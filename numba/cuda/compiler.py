import collections
import ctypes
import inspect
import os
import subprocess
import sys
import tempfile

import numpy as np

from numba import _dispatcher
from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate
from numba.core import (types, typing, utils, funcdesc, serialize, config,
                        compiler, sigutils)
from numba.core.typeconv.rules import default_type_manager
from numba.core.compiler_lock import global_compiler_lock
from numba.core.dispatcher import OmittedArg
from numba.core.typing.typeof import Purpose, typeof
import numba
from .cudadrv.devices import get_context
from .cudadrv import nvvm, driver
from .errors import missing_launch_config_msg, normalize_kernel_dimensions
from .api import get_current_device
from .args import wrap_arg


@global_compiler_lock
def compile_cuda(pyfunc, return_type, args, debug=False, inline=False):
    # First compilation will trigger the initialization of the CUDA backend.
    from .descriptor import CUDATargetDesc

    typingctx = CUDATargetDesc.typingctx
    targetctx = CUDATargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    flags.set('no_cpython_wrapper')
    flags.set('no_cfunc_wrapper')
    if debug:
        flags.set('debuginfo')
    if inline:
        flags.set('forceinline')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})

    library = cres.library
    library.finalize()

    return cres


@global_compiler_lock
def compile_kernel(pyfunc, args, link, debug=False, inline=False,
                   fastmath=False, extensions=[], max_registers=None, opt=True):
    cres = compile_cuda(pyfunc, types.void, args, debug=debug, inline=inline)
    fname = cres.fndesc.llvm_func_name
    lib, kernel = cres.target_context.prepare_cuda_kernel(cres.library, fname,
                                                          cres.signature.args,
                                                          debug=debug)

    cukern = _Kernel(llvm_module=lib._final_module,
                     name=kernel.name,
                     pretty_name=cres.fndesc.qualname,
                     argtypes=cres.signature.args,
                     type_annotation=cres.type_annotation,
                     link=link,
                     debug=debug,
                     opt=opt,
                     call_helper=cres.call_helper,
                     fastmath=fastmath,
                     extensions=extensions,
                     max_registers=max_registers)
    return cukern


@global_compiler_lock
def compile_ptx(pyfunc, args, debug=False, device=False, fastmath=False,
                cc=None, opt=True):
    """Compile a Python function to PTX for a given set of argument types.

    :param pyfunc: The Python function to compile.
    :param args: A tuple of argument types to compile for.
    :param debug: Whether to include debug info in the generated PTX.
    :type debug: bool
    :param device: Whether to compile a device function. Defaults to ``False``,
                   to compile global kernel functions.
    :type device: bool
    :param fastmath: Whether to enable fast math flags (ftz=1, prec_sqrt=0,
                     prec_div=, and fma=1)
    :type fastmath: bool
    :param cc: Compute capability to compile for, as a tuple ``(MAJOR, MINOR)``.
               Defaults to ``(5, 2)``.
    :type cc: tuple
    :param opt: Enable optimizations. Defaults to ``True``.
    :type opt: bool
    :return: (ptx, resty): The PTX code and inferred return type
    :rtype: tuple
    """
    cres = compile_cuda(pyfunc, None, args, debug=debug)
    resty = cres.signature.return_type
    if device:
        llvm_module = cres.library._final_module
        nvvm.fix_data_layout(llvm_module)
    else:
        fname = cres.fndesc.llvm_func_name
        tgt = cres.target_context
        lib, kernel = tgt.prepare_cuda_kernel(cres.library, fname,
                                              cres.signature.args, debug=debug)
        llvm_module = lib._final_module

    options = {
        'debug': debug,
        'fastmath': fastmath,
    }

    cc = cc or config.CUDA_DEFAULT_PTX_CC
    opt = 3 if opt else 0
    arch = nvvm.get_arch_option(*cc)
    llvmir = str(llvm_module)
    ptx = nvvm.llvm_to_ptx(llvmir, opt=opt, arch=arch, **options)
    return ptx.decode('utf-8'), resty


def compile_ptx_for_current_device(pyfunc, args, debug=False, device=False,
                                   fastmath=False, opt=True):
    """Compile a Python function to PTX for a given set of argument types for
    the current device's compute capabilility. This calls :func:`compile_ptx`
    with an appropriate ``cc`` value for the current device."""
    cc = get_current_device().compute_capability
    return compile_ptx(pyfunc, args, debug=-debug, device=device,
                       fastmath=fastmath, cc=cc, opt=True)


def disassemble_cubin(cubin):
    # nvdisasm only accepts input from a file, so we need to write out to a
    # temp file and clean up afterwards.
    fd = None
    fname = None
    try:
        fd, fname = tempfile.mkstemp()
        with open(fname, 'wb') as f:
            f.write(cubin)

        try:
            cp = subprocess.run(['nvdisasm', fname], check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            if e.filename == 'nvdisasm':
                msg = ("nvdisasm is required for SASS inspection, and has not "
                       "been found.\n\nYou may need to install the CUDA "
                       "toolkit and ensure that it is available on your "
                       "PATH.\n")
                raise RuntimeError(msg)
        return cp.stdout.decode('utf-8')
    finally:
        if fd is not None:
            os.close(fd)
        if fname is not None:
            os.unlink(fname)


class DeviceFunctionTemplate(serialize.ReduceMixin):
    """Unmaterialized device function
    """
    def __init__(self, pyfunc, debug, inline, opt):
        self.py_func = pyfunc
        self.debug = debug
        self.inline = inline
        self.opt = opt
        self._compileinfos = {}
        name = getattr(pyfunc, '__name__', 'unknown')
        self.__name__ = f"{name} <CUDA device function>".format(name)

    def _reduce_states(self):
        return dict(py_func=self.py_func, debug=self.debug, inline=self.inline)

    @classmethod
    def _rebuild(cls, py_func, debug, inline):
        return compile_device_template(py_func, debug=debug, inline=inline)

    def compile(self, args):
        """Compile the function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
        if args not in self._compileinfos:
            cres = compile_cuda(self.py_func, None, args, debug=self.debug,
                                inline=self.inline)
            first_definition = not self._compileinfos
            self._compileinfos[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(self, cres.fndesc,
                                                         libs)
            else:
                cres.target_context.add_user_function(self, cres.fndesc, libs)

        else:
            cres = self._compileinfos[args]

        return cres

    def inspect_llvm(self, args):
        """Returns the LLVM-IR text compiled for *args*.

        Parameters
        ----------
        args: tuple[Type]
            Argument types.

        Returns
        -------
        llvmir : str
        """
        # Force a compilation to occur if none has yet - this can be needed if
        # the user attempts to inspect LLVM IR or PTX before the function has
        # been called for the given arguments from a jitted kernel.
        self.compile(args)
        cres = self._compileinfos[args]
        mod = cres.library._final_module
        return str(mod)

    def inspect_ptx(self, args, nvvm_options={}):
        """Returns the PTX compiled for *args* for the currently active GPU

        Parameters
        ----------
        args: tuple[Type]
            Argument types.
        nvvm_options : dict; optional
            See `CompilationUnit.compile` in `numba/cuda/cudadrv/nvvm.py`.

        Returns
        -------
        ptx : bytes
        """
        llvmir = self.inspect_llvm(args)
        # Make PTX
        cuctx = get_context()
        device = cuctx.device
        cc = device.compute_capability
        arch = nvvm.get_arch_option(*cc)
        opt = 3 if self.opt else 0
        ptx = nvvm.llvm_to_ptx(llvmir, opt=opt, arch=arch, **nvvm_options)
        return ptx


def compile_device_template(pyfunc, debug=False, inline=False, opt=True):
    """Create a DeviceFunctionTemplate object and register the object to
    the CUDA typing context.
    """
    from .descriptor import CUDATargetDesc

    dft = DeviceFunctionTemplate(pyfunc, debug=debug, inline=inline, opt=opt)

    class device_function_template(AbstractTemplate):
        key = dft

        def generic(self, args, kws):
            assert not kws
            return dft.compile(args).signature

        def get_template_info(cls):
            basepath = os.path.dirname(os.path.dirname(numba.__file__))
            code, firstlineno = inspect.getsourcelines(pyfunc)
            path = inspect.getsourcefile(pyfunc)
            sig = str(utils.pysignature(pyfunc))
            info = {
                'kind': "overload",
                'name': getattr(cls.key, '__name__', "unknown"),
                'sig': sig,
                'filename': utils.safe_relpath(path, start=basepath),
                'lines': (firstlineno, firstlineno + len(code) - 1),
                'docstring': pyfunc.__doc__
            }
            return info

    typingctx = CUDATargetDesc.typingctx
    typingctx.insert_user_function(dft, device_function_template)
    return dft


def compile_device(pyfunc, return_type, args, inline=True, debug=False):
    return DeviceFunction(pyfunc, return_type, args, inline=True, debug=False)


def declare_device_function(name, restype, argtypes):
    from .descriptor import CUDATargetDesc

    typingctx = CUDATargetDesc.typingctx
    targetctx = CUDATargetDesc.targetctx
    sig = typing.signature(restype, *argtypes)
    extfn = ExternFunction(name, sig)

    class device_function_template(ConcreteTemplate):
        key = extfn
        cases = [sig]

    fndesc = funcdesc.ExternalFunctionDescriptor(
        name=name, restype=restype, argtypes=argtypes)
    typingctx.insert_user_function(extfn, device_function_template)
    targetctx.insert_user_function(extfn, fndesc)
    return extfn


class DeviceFunction(serialize.ReduceMixin):

    def __init__(self, pyfunc, return_type, args, inline, debug):
        self.py_func = pyfunc
        self.return_type = return_type
        self.args = args
        self.inline = True
        self.debug = False
        cres = compile_cuda(self.py_func, self.return_type, self.args,
                            debug=self.debug, inline=self.inline)
        self.cres = cres

        class device_function_template(ConcreteTemplate):
            key = self
            cases = [cres.signature]

        cres.typing_context.insert_user_function(
            self, device_function_template)
        cres.target_context.insert_user_function(self, cres.fndesc,
                                                 [cres.library])

    def _reduce_states(self):
        return dict(py_func=self.py_func, return_type=self.return_type,
                    args=self.args, inline=self.inline, debug=self.debug)

    @classmethod
    def _rebuild(cls, py_func, return_type, args, inline, debug):
        return cls(py_func, return_type, args, inline, debug)

    def __repr__(self):
        fmt = "<DeviceFunction py_func={0} signature={1}>"
        return fmt.format(self.py_func, self.cres.signature)


class ExternFunction(object):
    def __init__(self, name, sig):
        self.name = name
        self.sig = sig


class ForAll(object):
    def __init__(self, kernel, ntasks, tpb, stream, sharedmem):
        if ntasks < 0:
            raise ValueError("Can't create ForAll with negative task count: %s"
                             % ntasks)
        self.kernel = kernel
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if self.ntasks == 0:
            return

        if self.kernel.specialized:
            kernel = self.kernel
        else:
            kernel = self.kernel.specialize(*args)
        blockdim = self._compute_thread_per_block(kernel)
        griddim = (self.ntasks + blockdim - 1) // blockdim

        return kernel[griddim, blockdim, self.stream, self.sharedmem](*args)

    def _compute_thread_per_block(self, kernel):
        tpb = self.thread_per_block
        # Prefer user-specified config
        if tpb != 0:
            return tpb
        # Else, ask the driver to give a good config
        else:
            ctx = get_context()
            kwargs = dict(
                func=kernel._func.get(),
                b2d_func=0,     # dynamic-shared memory is constant to blksz
                memsize=self.sharedmem,
                blocksizelimit=1024,
            )
            _, tpb = ctx.get_max_potential_block_size(**kwargs)
            return tpb


class CachedPTX(object):
    """A PTX cache that uses compute capability as a cache key
    """
    def __init__(self, name, llvmir, options):
        self.name = name
        self.llvmir = llvmir
        self.cache = {}
        self._extra_options = options.copy()

    def get(self):
        """
        Get PTX for the current active context.
        """
        cuctx = get_context()
        device = cuctx.device
        cc = device.compute_capability
        ptx = self.cache.get(cc)
        if ptx is None:
            arch = nvvm.get_arch_option(*cc)
            ptx = nvvm.llvm_to_ptx(self.llvmir, arch=arch,
                                   **self._extra_options)
            self.cache[cc] = ptx
            if config.DUMP_ASSEMBLY:
                print(("ASSEMBLY %s" % self.name).center(80, '-'))
                print(ptx.decode('utf-8'))
                print('=' * 80)
        return ptx


class CachedCUFunction(serialize.ReduceMixin):
    """
    Get or compile CUDA function for the current active context

    Uses device ID as key for cache.
    """

    def __init__(self, entry_name, ptx, linking, max_registers):
        self.entry_name = entry_name
        self.ptx = ptx
        self.linking = linking
        self.cache = {}
        self.ccinfos = {}
        self.cubins = {}
        self.max_registers = max_registers

    def get(self):
        cuctx = get_context()
        device = cuctx.device
        cufunc = self.cache.get(device.id)
        if cufunc is None:
            ptx = self.ptx.get()

            # Link
            linker = driver.Linker(max_registers=self.max_registers)
            linker.add_ptx(ptx)
            for path in self.linking:
                linker.add_file_guess_ext(path)
            cubin, size = linker.complete()
            compile_info = linker.info_log
            module = cuctx.create_module_image(cubin)

            # Load
            cufunc = module.get_function(self.entry_name)

            # Populate caches
            self.cache[device.id] = cufunc
            self.ccinfos[device.id] = compile_info
            # We take a copy of the cubin because it's owned by the linker
            cubin_ptr = ctypes.cast(cubin, ctypes.POINTER(ctypes.c_char))
            cubin_data = np.ctypeslib.as_array(cubin_ptr, shape=(size,)).copy()
            self.cubins[device.id] = cubin_data
        return cufunc

    def get_sass(self):
        self.get()  # trigger compilation
        device = get_context().device
        return disassemble_cubin(self.cubins[device.id])

    def get_info(self):
        self.get()   # trigger compilation
        cuctx = get_context()
        device = cuctx.device
        ci = self.ccinfos[device.id]
        return ci

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Pre-compiled PTX code string is serialized inside the `ptx` (CachedPTX).
        Loaded CUfunctions are discarded. They are recreated when unserialized.
        """
        if self.linking:
            msg = ('cannot pickle CUDA kernel function with additional '
                   'libraries to link against')
            raise RuntimeError(msg)
        return dict(entry_name=self.entry_name, ptx=self.ptx,
                    linking=self.linking, max_registers=self.max_registers)

    @classmethod
    def _rebuild(cls, entry_name, ptx, linking, max_registers):
        """
        Rebuild an instance.
        """
        return cls(entry_name, ptx, linking, max_registers)


class _Kernel(serialize.ReduceMixin):
    '''
    CUDA Kernel specialized for a given set of argument types. When called, this
    object launches the kernel on the device.
    '''
    def __init__(self, llvm_module, name, pretty_name, argtypes, call_helper,
                 link=(), debug=False, fastmath=False, type_annotation=None,
                 extensions=[], max_registers=None, opt=True):
        super().__init__()
        # initialize CUfunction
        options = {
            'debug': debug,
            'fastmath': fastmath,
            'opt': 3 if opt else 0
        }

        ptx = CachedPTX(pretty_name, str(llvm_module), options=options)
        cufunc = CachedCUFunction(name, ptx, link, max_registers)
        # populate members
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.linking = tuple(link)
        self._type_annotation = type_annotation
        self._func = cufunc
        self.debug = debug
        self.call_helper = call_helper
        self.extensions = list(extensions)

    @classmethod
    def _rebuild(cls, name, argtypes, cufunc, link, debug, call_helper,
                 extensions):
        """
        Rebuild an instance.
        """
        instance = cls.__new__(cls)
        # invoke parent constructor
        super(cls, instance).__init__()
        # populate members
        instance.entry_name = name
        instance.argument_types = tuple(argtypes)
        instance.linking = tuple(link)
        instance._type_annotation = None
        instance._func = cufunc
        instance.debug = debug
        instance.call_helper = call_helper
        instance.extensions = extensions
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are serialized in PTX form.
        Type annotation are discarded.
        Thread, block and shared memory configuration are serialized.
        Stream information is discarded.
        """
        return dict(name=self.entry_name, argtypes=self.argument_types,
                    cufunc=self._func, link=self.linking, debug=self.debug,
                    call_helper=self.call_helper, extensions=self.extensions)

    def __call__(self, *args, **kwargs):
        assert not kwargs
        griddim, blockdim = normalize_kernel_dimensions(self.griddim,
                                                        self.blockdim)
        self._kernel_call(args=args,
                          griddim=griddim,
                          blockdim=blockdim,
                          stream=self.stream,
                          sharedmem=self.sharedmem)

    def bind(self):
        """
        Force binding to current CUDA context
        """
        self._func.get()

    @property
    def ptx(self):
        '''
        PTX code for this kernel.
        '''
        return self._func.ptx.get().decode('utf8')

    @property
    def device(self):
        """
        Get current active context
        """
        return get_current_device()

    def inspect_llvm(self):
        '''
        Returns the LLVM IR for this kernel.
        '''
        return str(self._func.ptx.llvmir)

    def inspect_asm(self):
        '''
        Returns the PTX code for this kernel.
        '''
        return self._func.ptx.get().decode('ascii')

    def inspect_sass(self):
        '''
        Returns the SASS code for this kernel.

        Requires nvdisasm to be available on the PATH.
        '''
        return self._func.get_sass()

    def inspect_types(self, file=None):
        '''
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        '''
        if self._type_annotation is None:
            raise ValueError("Type annotation is not available")

        if file is None:
            file = sys.stdout

        print("%s %s" % (self.entry_name, self.argument_types), file=file)
        print('-' * 80, file=file)
        print(self._type_annotation, file=file)
        print('=' * 80, file=file)

    def launch(self, args, griddim, blockdim, stream=0, sharedmem=0):
        # Prepare kernel
        cufunc = self._func.get()

        if self.debug:
            excname = cufunc.name + "__errcode__"
            excmem, excsz = cufunc.module.get_global_symbol(excname)
            assert excsz == ctypes.sizeof(ctypes.c_int)
            excval = ctypes.c_int()
            excmem.memset(0, stream=stream)

        # Prepare arguments
        retr = []                       # hold functors for writeback

        kernelargs = []
        for t, v in zip(self.argument_types, args):
            self._prepare_args(t, v, stream, retr, kernelargs)

        # Configure kernel
        cu_func = cufunc.configure(griddim, blockdim,
                                   stream=stream,
                                   sharedmem=sharedmem)
        # Invoke kernel
        cu_func(*kernelargs)

        if self.debug:
            driver.device_to_host(ctypes.addressof(excval), excmem, excsz)
            if excval.value != 0:
                # An error occurred
                def load_symbol(name):
                    mem, sz = cufunc.module.get_global_symbol("%s__%s__" %
                                                              (cufunc.name,
                                                               name))
                    val = ctypes.c_int()
                    driver.device_to_host(ctypes.addressof(val), mem, sz)
                    return val.value

                tid = [load_symbol("tid" + i) for i in 'zyx']
                ctaid = [load_symbol("ctaid" + i) for i in 'zyx']
                code = excval.value
                exccls, exc_args, loc = self.call_helper.get_exception(code)
                # Prefix the exception message with the source location
                if loc is None:
                    locinfo = ''
                else:
                    sym, filepath, lineno = loc
                    filepath = os.path.abspath(filepath)
                    locinfo = 'In function %r, file %s, line %s, ' % (sym,
                                                                      filepath,
                                                                      lineno,)
                # Prefix the exception message with the thread position
                prefix = "%stid=%s ctaid=%s" % (locinfo, tid, ctaid)
                if exc_args:
                    exc_args = ("%s: %s" % (prefix, exc_args[0]),) + \
                        exc_args[1:]
                else:
                    exc_args = prefix,
                raise exccls(*exc_args)

        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """

        # map the arguments using any extension you've registered
        for extension in reversed(self.extensions):
            ty, val = extension.prepare_args(
                ty,
                val,
                stream=stream,
                retr=retr)

        if isinstance(ty, types.Array):
            devary = wrap_arg(val).to_device(retr, stream)

            c_intp = ctypes.c_ssize_t

            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)
            data = ctypes.c_void_p(driver.device_pointer(devary))
            kernelargs.append(meminfo)
            kernelargs.append(parent)
            kernelargs.append(nitems)
            kernelargs.append(itemsize)
            kernelargs.append(data)
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.shape[ax]))
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.strides[ax]))

        elif isinstance(ty, types.Integer):
            cval = getattr(ctypes, "c_%s" % ty)(val)
            kernelargs.append(cval)

        elif ty == types.float64:
            cval = ctypes.c_double(val)
            kernelargs.append(cval)

        elif ty == types.float32:
            cval = ctypes.c_float(val)
            kernelargs.append(cval)

        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            kernelargs.append(cval)

        elif ty == types.complex64:
            kernelargs.append(ctypes.c_float(val.real))
            kernelargs.append(ctypes.c_float(val.imag))

        elif ty == types.complex128:
            kernelargs.append(ctypes.c_double(val.real))
            kernelargs.append(ctypes.c_double(val.imag))

        elif isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
            kernelargs.append(ctypes.c_int64(val.view(np.int64)))

        elif isinstance(ty, types.Record):
            devrec = wrap_arg(val).to_device(retr, stream)
            kernelargs.append(devrec)

        else:
            raise NotImplementedError(ty, val)


class _KernelConfiguration:
    def __init__(self, dispatcher, griddim, blockdim, stream, sharedmem):
        self.dispatcher = dispatcher
        self.griddim = griddim
        self.blockdim = blockdim
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        return self.dispatcher.call(args, self.griddim, self.blockdim,
                                    self.stream, self.sharedmem)


class Dispatcher(_dispatcher.Dispatcher, serialize.ReduceMixin):
    '''
    CUDA Dispatcher object. When configured and called, the dispatcher will
    specialize itself for the given arguments (if no suitable specialized
    version already exists) & compute capability, and launch on the device
    associated with the current context.

    Dispatcher objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    '''

    # Whether to fold named arguments and default values. These are presently
    # not supported on CUDA IIRC.
    _fold_args = False

    def __init__(self, py_func, sigs, targetoptions):
        self.py_func = py_func
        self.sigs = []
        self.link = targetoptions.pop('link', (),)
        self._can_compile = True

        # keyed by a `(compute capability, args)` tuple
        self.definitions = {}
        self.specializations = {}

        self.targetoptions = targetoptions

        # defensive copy
        self.targetoptions['extensions'] = \
            list(self.targetoptions.get('extensions', []))

        from .descriptor import CUDATargetDesc

        self.typingctx = CUDATargetDesc.typingctx

        self._tm = default_type_manager

        pysig = utils.pysignature(py_func)
        arg_count = len(pysig.parameters)
        argnames = tuple(pysig.parameters)
        default_values = self.py_func.__defaults__ or ()
        defargs = tuple(OmittedArg(val) for val in default_values)
        can_fallback = False # Fallback to object mode?

        try:
            lastarg = list(pysig.parameters.values())[-1]
        except IndexError:
            has_stararg = False
        else:
            has_stararg = lastarg.kind == lastarg.VAR_POSITIONAL

        # Not clear when this is ever true anywhere in Numba
        exact_match_required = False

        # A mapping of signatures to compile results
        # Stopgap for _DispatcherBase
        self.overloads = collections.OrderedDict()

        _dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(),
                                        arg_count, self._fold_args, argnames,
                                        defargs, can_fallback, has_stararg,
                                        exact_match_required)

        if sigs:
            if len(sigs) > 1:
                raise TypeError("Only one signature supported at present")
            self.compile(sigs[0])
            self._can_compile = False

    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)
        return _KernelConfiguration(self, griddim, blockdim, stream, sharedmem)

    def __getitem__(self, args):
        if len(args) not in [2, 3, 4]:
            raise ValueError('must specify at least the griddim and blockdim')
        return self.configure(*args)

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        """Returns a 1D-configured kernel for a given number of tasks.

        This assumes that:

        - the kernel maps the Global Thread ID ``cuda.grid(1)`` to tasks on a
          1-1 basis.
        - the kernel checks that the Global Thread ID is upper-bounded by
          ``ntasks``, and does nothing if it is not.

        :param ntasks: The number of tasks.
        :param tpb: The size of a block. An appropriate value is chosen if this
                    parameter is not supplied.
        :param stream: The stream on which the configured kernel will be
                       launched.
        :param sharedmem: The number of bytes of dynamic shared memory required
                          by the kernel.
        :return: A configured kernel, ready to launch on a set of arguments."""

        return ForAll(self, ntasks, tpb=tpb, stream=stream, sharedmem=sharedmem)

    @property
    def extensions(self):
        '''
        A list of objects that must have a `prepare_args` function. When a
        specialized kernel is called, each argument will be passed through
        to the `prepare_args` (from the last object in this list to the
        first). The arguments to `prepare_args` are:

        - `ty` the numba type of the argument
        - `val` the argument value itself
        - `stream` the CUDA stream used for the current call to the kernel
        - `retr` a list of zero-arg functions that you may want to append
          post-call cleanup work to.

        The `prepare_args` function must return a tuple `(ty, val)`, which
        will be passed in turn to the next right-most `extension`. After all
        the extensions have been called, the resulting `(ty, val)` will be
        passed into Numba's default argument marshalling logic.
        '''
        return self.targetoptions['extensions']

    def __call__(self, *args, **kwargs):
        # An attempt to launch an unconfigured kernel
        raise ValueError(missing_launch_config_msg)

    def call(self, args, griddim, blockdim, stream, sharedmem):
        '''
        Compile if necessary and invoke this kernel with *args*.
        '''
        #argtypes = tuple(
        #    [self.typingctx.resolve_argument_type(a) for a in args])
        kernel = _dispatcher.Dispatcher._cuda_call(self, *args)
        #kernel = self.compile(argtypes)
        kernel.launch(args, griddim, blockdim, stream, sharedmem)

    def _compile_for_args(self, *args, **kws):
        # Stopgap. Ideally we would move to using
        # _DispatcherBase._compile_for_args.
        assert not kws
        argtypes = [typeof(a, Purpose.argument) for a in args]
        return self.compile(tuple(argtypes))

    def _search_new_conversions(self, *args, **kws):
        # Stopgap. need to move to using
        # _DispatcherBase._search_new_conversions
        assert not kws
        args = [typeof(a, Purpose.argument) for a in args]
        found = False
        for sig in self.nopython_signatures:
            conv = self.typingctx.install_possible_conversions(args, sig.args)
            if conv:
                found = True
        return found

    def typeof_pyval(self, val):
        # Stopgap. Need to move to _DispatcherBase.typeof_pyval
        return typeof(val, Purpose.argument)

    @property
    def nopython_signatures(self):
        # Stopgap from _DispatcherBase
        return [cres.signature for cres in self.overloads.values()
                if not cres.objectmode and not cres.interpmode]

    def specialize(self, *args):
        '''
        Create a new instance of this dispatcher specialized for the given
        *args*.
        '''
        cc = get_current_device().compute_capability
        argtypes = tuple(
            [self.typingctx.resolve_argument_type(a) for a in args])
        if self.specialized:
            raise RuntimeError('Dispatcher already specialized')

        specialization = self.specializations.get((cc, argtypes))
        if specialization:
            return specialization

        targetoptions = self.targetoptions
        targetoptions['link'] = self.link
        specialization = Dispatcher(self.py_func, [types.void(*argtypes)],
                                    targetoptions)
        self.specializations[cc, argtypes] = specialization
        return specialization

    def disable_compile(self, val=True):
        self._can_compile = not val

    @property
    def specialized(self):
        """
        True if the Dispatcher has been specialized.
        """
        return len(self.sigs) == 1 and not self._can_compile

    @property
    def definition(self):
        # There is a single definition only when the dispatcher has been
        # specialized.
        if not self.specialized:
            raise ValueError("Dispatcher needs to be specialized to get the "
                             "single definition")
        return next(iter(self.definitions.values()))

    @property
    def _func(self, signature=None, compute_capability=None):
        cc = compute_capability or get_current_device().compute_capability
        if signature is not None:
            return self.definitions[(cc, signature)]._func
        elif self.specialized:
            return self.definition._func
        else:
            return {sig: defn._func for sig, defn in self.definitions.items()}

    def compile(self, sig):
        '''
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        '''
        # Need to add overload here, I think. Not use self.definitions anymore.
        print("Compiling")
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None or return_type == types.none
        cc = get_current_device().compute_capability
        if self.specialized:
            return self.definition
        else:
            kernel = self.definitions.get((cc, argtypes))
        if kernel is None:
            if not self._can_compile:
                raise RuntimeError("Compilation disabled")
            kernel = compile_kernel(self.py_func, argtypes,
                                    link=self.link,
                                    **self.targetoptions)
            self.definitions[(cc, argtypes)] = kernel
            # Inspired by _DispatcherBase.add_overload - another Stopgap.
            #from pudb import set_trace; set_trace()
            # Seems surprising that we sometimes get a tuple and sometimes a
            # signature. Looks like we could do with some better normalization
            # somewhere.
            if isinstance(sig, tuple):
                c_sig = [a._code for a in sig]
            else:
                c_sig = [a._code for a in sig.args]
            self._cuda_insert(c_sig, kernel)
            self.overloads[argtypes] = kernel

            kernel.bind()
            self.sigs.append(sig)
        return kernel

    def inspect_llvm(self, signature=None, compute_capability=None):
        '''
        Return the LLVM IR for all signatures encountered thus far, or the LLVM
        IR for a specific signature and compute_capability if given. If the
        dispatcher is specialized, the IR for the single specialization is
        returned.
        '''
        cc = compute_capability or get_current_device().compute_capability
        if signature is not None:
            return self.definitions[(cc, signature)].inspect_llvm()
        elif self.specialized:
            return self.definition.inspect_llvm()
        else:
            return dict((sig, defn.inspect_llvm())
                        for sig, defn in self.definitions.items())

    def inspect_asm(self, signature=None, compute_capability=None):
        '''
        Return the generated PTX assembly code for all signatures encountered
        thus far, or the PTX assembly code for a specific signature and
        compute_capability if given. If the dispatcher is specialized, the
        assembly code for the single specialization is returned.
        '''
        cc = compute_capability or get_current_device().compute_capability
        if signature is not None:
            return self.definitions[(cc, signature)].inspect_asm()
        elif self.specialized:
            return self.definition.inspect_asm()
        else:
            return dict((sig, defn.inspect_asm())
                        for sig, defn in self.definitions.items())

    def inspect_sass(self, signature=None, compute_capability=None):
        '''
        Return the generated SASS code for all signatures encountered thus
        far, or the SASS code for a specific signature and compute_capability
        if given.

        Requires nvdisasm to be available on the PATH.
        '''
        cc = compute_capability or get_current_device().compute_capability
        if signature is not None:
            return self.definitions[(cc, signature)].inspect_sass()
        elif self.specialized:
            return self.definition.inspect_sass()
        else:
            return dict((sig, defn.inspect_sass())
                        for sig, defn in self.definitions.items())

    def inspect_types(self, file=None):
        '''
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        '''
        if file is None:
            file = sys.stdout

        if self.specialized:
            self.definition.inspect_types(file=file)
        else:
            for _, defn in self.definitions.items():
                defn.inspect_types(file=file)

    @property
    def ptx(self):
        if self.specialized:
            return self.definition.ptx
        else:
            return dict((sig, defn.ptx)
                        for sig, defn in self.definitions.items())

    def bind(self):
        for defn in self.definitions.values():
            defn.bind()

    @classmethod
    def _rebuild(cls, py_func, sigs, targetoptions):
        """
        Rebuild an instance.
        """
        instance = cls(py_func, sigs, targetoptions)
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are discarded.
        """
        return dict(py_func=self.py_func, sigs=self.sigs,
                    targetoptions=self.targetoptions)
