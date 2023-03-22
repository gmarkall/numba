from llvmlite import ir
from numba.core.typing.templates import ConcreteTemplate
from numba.core import (itanium_mangler, types, typing, funcdesc, config,
                        compiler, sigutils, cgutils)
from numba.core.compiler import (sanitize_compile_result_entries, CompilerBase,
                                 DefaultPassBuilder, Flags, Option,
                                 CompileResult)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass, AnalysisPass,
                                           PassManager, register_pass)
from numba.core.errors import NumbaInvalidConfigWarning, TypingError
from numba.core.typed_passes import (IRLegalization, NativeLowering,
                                     AnnotateTypes)
from warnings import warn
from numba.cuda.api import get_current_device
from numba.cuda import nvvmutils


def _nvvm_options_type(x):
    if x is None:
        return None

    else:
        assert isinstance(x, dict)
        return x


class CUDAFlags(Flags):
    nvvm_options = Option(
        type=_nvvm_options_type,
        default=None,
        doc="NVVM options",
    )
    compute_capability = Option(
        type=tuple,
        default=None,
        doc="Compute Capability",
    )


# The CUDACompileResult (CCR) has a specially-defined entry point equal to its
# id.  This is because the entry point is used as a key into a dict of
# overloads by the base dispatcher. The id of the CCR is the only small and
# unique property of a CompileResult in the CUDA target (cf. the CPU target,
# which uses its entry_point, which is a pointer value).
#
# This does feel a little hackish, and there are two ways in which this could
# be improved:
#
# 1. We could change the core of Numba so that each CompileResult has its own
#    unique ID that can be used as a key - e.g. a count, similar to the way in
#    which types have unique counts.
# 2. At some future time when kernel launch uses a compiled function, the entry
#    point will no longer need to be a synthetic value, but will instead be a
#    pointer to the compiled function as in the CPU target.

class CUDACompileResult(CompileResult):
    @property
    def entry_point(self):
        return id(self)


def cuda_compile_result(**entries):
    entries = sanitize_compile_result_entries(entries)
    return CUDACompileResult(**entries)


@register_pass(mutates_CFG=True, analysis_only=False)
class CUDABackend(LoweringPass):

    _name = "cuda_backend"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Packages lowering output in a compile result
        """
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)

        flags = state.flags
        debug = flags.debuginfo and not flags.dbg_directives_only
        lineinfo = flags.debuginfo and flags.dbg_directives_only
        loc = state.func_ir.loc
        exceptions = False # XXX: Need to have it passed through somehow
        prepare_cuda_kernel(state.targetctx, state.library, lowered.fndesc,
                            debug, lineinfo, exceptions, loc.filename,
                            loc.line)
        state.cr = cuda_compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            fndesc=lowered.fndesc,
        )
        return True


@register_pass(mutates_CFG=False, analysis_only=False)
class CreateLibrary(LoweringPass):
    """
    Create a CUDACodeLibrary for the NativeLowering pass to populate. The
    NativeLowering pass will create a code library if none exists, but we need
    to set it up with nvvm_options from the flags if they are present.
    """

    _name = "create_library"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        codegen = state.targetctx.codegen()
        name = state.func_id.func_qualname
        nvvm_options = state.flags.nvvm_options
        state.library = codegen.create_library(name, nvvm_options=nvvm_options)
        # Enable object caching upfront so that the library can be serialized.
        state.library.enable_object_caching()

        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class CUDALegalization(AnalysisPass):

    _name = "cuda_legalization"

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        # Early return if NVVM 7
        from numba.cuda.cudadrv.nvvm import NVVM
        if NVVM().is_nvvm70:
            return False
        # NVVM < 7, need to check for charseq
        typmap = state.typemap

        def check_dtype(dtype):
            if isinstance(dtype, (types.UnicodeCharSeq, types.CharSeq)):
                msg = (f"{k} is a char sequence type. This type is not "
                       "supported with CUDA toolkit versions < 11.2. To "
                       "use this type, you need to update your CUDA "
                       "toolkit - try 'conda install cudatoolkit=11' if "
                       "you are using conda to manage your environment.")
                raise TypingError(msg)
            elif isinstance(dtype, types.Record):
                for subdtype in dtype.fields.items():
                    # subdtype is a (name, _RecordField) pair
                    check_dtype(subdtype[1].type)

        for k, v in typmap.items():
            if isinstance(v, types.Array):
                check_dtype(v.dtype)
        return False


class CUDACompiler(CompilerBase):
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('cuda')

        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)
        pm.add_pass(CUDALegalization, "CUDA legalization")

        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]

    def define_cuda_lowering_pipeline(self, state):
        pm = PassManager('cuda_lowering')
        # legalise
        pm.add_pass(IRLegalization,
                    "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")

        # lower
        pm.add_pass(CreateLibrary, "create library")
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(CUDABackend, "cuda backend")

        pm.finalize()
        return pm


def set_cuda_kernel(func):
    mod = func.module

    mdstr = ir.MetaDataString(mod, "kernel")
    mdvalue = ir.Constant(ir.IntType(32), 1)
    md = mod.add_metadata((func, mdstr, mdvalue))

    nmd = cgutils.get_or_insert_named_metadata(mod, 'nvvm.annotations')
    nmd.add(md)

    # Marking a kernel 'noinline' causes NVVM to generate a warning, so remove
    # it if it is present.
    func.attributes.discard('noinline')


def prepare_cuda_kernel(context, library, fndesc, debug, lineinfo, exceptions,
                        filename, linenum, max_registers=None):
    """
    Adapt a code library ``codelib`` with the numba compiled CUDA kernel
    with name ``fname`` and arguments ``argtypes`` for NVVM.
    A new library is created with a wrapper function that can be used as
    the kernel entry point for the given kernel.

    Returns the new code library and the wrapper function.

    Parameters:

    context:       The target context
    library:       The CodeLibrary containing the device function to wrap
                   in a kernel call.
    fndesc:        The FunctionDescriptor of the source function.
    debug:         Whether to compile with debug.
    lineinfo:      Whether to emit line info.
    filename:      The source filename that the function is contained in.
    linenum:       The source line that the function is on.
    max_registers: The max_registers argument for the code library.
    """
    kernel_name = itanium_mangler.prepend_namespace(
        fndesc.llvm_func_name, ns='cudapy',
    )
    library._entry_name = kernel_name

    argtypes = fndesc.argtypes
    arginfo = context.get_arg_packer(argtypes)
    argtys = list(arginfo.argument_types)
    wrapfnty = ir.FunctionType(ir.VoidType(), argtys)
    wrapper_module = library._module
    for func in wrapper_module.functions:
        if func.name == fndesc.llvm_func_name:
            break

    wrapfn = ir.Function(wrapper_module, wrapfnty, kernel_name)
    builder = ir.IRBuilder(wrapfn.append_basic_block(''))

    if debug or lineinfo:
        directives_only = lineinfo and not debug
        debuginfo = context.DIBuilder(module=wrapper_module,
                                      filepath=filename,
                                      cgctx=context,
                                      directives_only=directives_only)
        debuginfo.mark_subprogram(
            wrapfn, kernel_name, fndesc.args, argtypes, linenum,
        )
        debuginfo.mark_location(builder, linenum)

    callargs = arginfo.from_arguments(builder, wrapfn.args)
    status, _ = context.call_conv.call_function(
        builder, func, types.void, argtypes, callargs)

    if exceptions:
        generate_exception_check(builder, wrapfn.name, status)

    builder.ret_void()

    set_cuda_kernel(wrapfn)

    if debug or lineinfo:
        debuginfo.finalize()


def generate_exception_check(builder, name, status):
    # Define error handling variable
    def define_error_gv(postfix):
        gv = cgutils.add_global_variable(builder.module, ir.IntType(32),
                                         f'{name}{postfix}')
        gv.initializer = ir.Constant(gv.type.pointee, None)
        return gv

    gv_exc = define_error_gv("__errcode__")
    gv_tid = []
    gv_ctaid = []
    for i in 'xyz':
        gv_tid.append(define_error_gv("__tid%s__" % i))
        gv_ctaid.append(define_error_gv("__ctaid%s__" % i))

    # Check error status
    with cgutils.if_likely(builder, status.is_ok):
        builder.ret_void()

    with builder.if_then(builder.not_(status.is_python_exc)):
        # User exception raised
        old = ir.Constant(gv_exc.type.pointee, None)

        # Use atomic cmpxchg to prevent rewriting the error status
        # Only the first error is recorded
        xchg = builder.cmpxchg(gv_exc, old, status.code,
                               'monotonic', 'monotonic')
        changed = builder.extract_value(xchg, 1)

        # If the xchange is successful, save the thread ID.
        sreg = nvvmutils.SRegBuilder(builder)
        with builder.if_then(changed):
            for dim, ptr, in zip("xyz", gv_tid):
                val = sreg.tid(dim)
                builder.store(val, ptr)

            for dim, ptr, in zip("xyz", gv_ctaid):
                val = sreg.ctaid(dim)
                builder.store(val, ptr)


@global_compiler_lock
def compile_cuda(pyfunc, return_type, args, debug=False, lineinfo=False,
                 exceptions=False, inline=False, fastmath=False,
                 nvvm_options=None, cc=None):
    if cc is None:
        raise ValueError('Compute Capability must be supplied')

    from .descriptor import cuda_target
    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context

    flags = CUDAFlags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True

    # Both debug and lineinfo turn on debug information in the compiled code,
    # but we keep them separate arguments in case we later want to overload
    # some other behavior on the debug flag. In particular, -opt=3 is not
    # supported with debug enabled, and enabling only lineinfo should not
    # affect the error model.
    if debug or lineinfo:
        flags.debuginfo = True

    if lineinfo:
        flags.dbg_directives_only = True

    if exceptions:
        flags.error_model = 'python'
    else:
        flags.error_model = 'numpy'

    if inline:
        flags.forceinline = True
    if fastmath:
        flags.fastmath = True
    if nvvm_options:
        flags.nvvm_options = nvvm_options
    flags.compute_capability = cc

    # Run compilation pipeline
    from numba.core.target_extension import target_override
    with target_override('cuda'):
        cres = compiler.compile_extra(typingctx=typingctx,
                                      targetctx=targetctx,
                                      func=pyfunc,
                                      args=args,
                                      return_type=return_type,
                                      flags=flags,
                                      locals={},
                                      pipeline_class=CUDACompiler)

    library = cres.library
    library.finalize()

    return cres


@global_compiler_lock
def compile_ptx(pyfunc, sig, debug=False, lineinfo=False, exceptions=False,
                device=False, fastmath=False, cc=None, opt=True):
    """Compile a Python function to PTX for a given set of argument types.

    :param pyfunc: The Python function to compile.
    :param sig: The signature representing the function's input and output
                types.
    :param debug: Whether to include debug info in the generated PTX.
    :type debug: bool
    :param lineinfo: Whether to include a line mapping from the generated PTX
                     to the source code. Usually this is used with optimized
                     code (since debug mode would automatically include this),
                     so we want debug info in the LLVM but only the line
                     mapping in the final PTX.
    :type lineinfo: bool
    :param exceptions: Whether the generated code should check for exceptions
                       during its execution. Exceptions are reported according
                       to Numba's CUDA Calling Convention.
    :type exceptions: bool
    :param device: Whether to compile a device function. Defaults to ``False``,
                   to compile global kernel functions.
    :type device: bool
    :param fastmath: Whether to enable fast math flags (ftz=1, prec_sqrt=0,
                     prec_div=, and fma=1)
    :type fastmath: bool
    :param cc: Compute capability to compile for, as a tuple
               ``(MAJOR, MINOR)``. Defaults to ``(5, 3)``.
    :type cc: tuple
    :param opt: Enable optimizations. Defaults to ``True``.
    :type opt: bool
    :return: (ptx, resty): The PTX code and inferred return type
    :rtype: tuple
    """
    if debug and opt:
        msg = ("debug=True with opt=True (the default) "
               "is not supported by CUDA. This may result in a crash"
               " - set debug=False or opt=False.")
        warn(NumbaInvalidConfigWarning(msg))

    nvvm_options = {
        'fastmath': fastmath,
        'opt': 3 if opt else 0
    }

    args, return_type = sigutils.normalize_signature(sig)

    cc = cc or config.CUDA_DEFAULT_PTX_CC
    cres = compile_cuda(pyfunc, return_type, args, debug=debug,
                        lineinfo=lineinfo, exceptions=exceptions,
                        fastmath=fastmath, nvvm_options=nvvm_options, cc=cc)
    resty = cres.signature.return_type

    if resty and not device and resty != types.void:
        raise TypeError("CUDA kernel must have void return type.")

    ptx = cres.library.get_asm_str(cc=cc)
    return ptx, resty


def compile_ptx_for_current_device(pyfunc, sig, debug=False, lineinfo=False,
                                   device=False, fastmath=False, opt=True):
    """Compile a Python function to PTX for a given set of argument types for
    the current device's compute capabilility. This calls :func:`compile_ptx`
    with an appropriate ``cc`` value for the current device."""
    cc = get_current_device().compute_capability
    return compile_ptx(pyfunc, sig, debug=debug, lineinfo=lineinfo,
                       device=device, fastmath=fastmath, cc=cc, opt=True)


def declare_device_function(name, restype, argtypes):
    return declare_device_function_template(name, restype, argtypes).key


def declare_device_function_template(name, restype, argtypes):
    from .descriptor import cuda_target
    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context
    sig = typing.signature(restype, *argtypes)
    extfn = ExternFunction(name, sig)

    class device_function_template(ConcreteTemplate):
        key = extfn
        cases = [sig]

    fndesc = funcdesc.ExternalFunctionDescriptor(
        name=name, restype=restype, argtypes=argtypes)
    typingctx.insert_user_function(extfn, device_function_template)
    targetctx.insert_user_function(extfn, fndesc)

    return device_function_template


class ExternFunction(object):
    def __init__(self, name, sig):
        self.name = name
        self.sig = sig
