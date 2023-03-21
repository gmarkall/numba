from warnings import warn
from numba.core import config, sigutils
from numba.core.errors import DeprecationError, NumbaInvalidConfigWarning
from numba.cuda.compiler import declare_device_function
from numba.cuda.dispatcher import CUDADispatcher
from numba.cuda.simulator.kernel import FakeCUDAKernel


_msg_deprecated_signature_arg = ("Deprecated keyword argument `{0}`. "
                                 "Signatures should be passed as the first "
                                 "positional argument.")


def jit(func_or_sig=None, device=False, cache=False, extensions=None,
        **options):
    """
    JIT compile a Python function for CUDA GPUs.

    :param func_or_sig: A function to JIT compile, or *signatures* of a
       function to compile. If a function is supplied, then a
       :class:`Dispatcher <numba.cuda.dispatcher.CUDADispatcher>` is returned.
       Otherwise, ``func_or_sig`` may be a signature or a list of signatures,
       and a function is returned. The returned function accepts another
       function, which it will compile and then return a :class:`Dispatcher
       <numba.cuda.dispatcher.CUDADispatcher>`. See :ref:`jit-decorator` for
       more information about passing signatures.

       .. note:: A kernel cannot have any return value.
    :param device: Indicates whether this is a device function.
    :type device: bool
    :param link: A list of files containing PTX or CUDA C/C++ source to link
       with the function
    :type link: list
    :param debug: If True, debug info is generated for compiled kernels. Since
       this degrades performance, this should only be used for debugging
       purposes.  If set to True, then ``opt`` should be set to False.
       Defaults to False.  (The default value can be overridden by setting
       environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
    :param exceptions: If True, check for exceptions thrown when executing
       the kernel. Since this degrades performance, this should only be used
       for debugging purposes. Defaults to False.  (The default value can be
       overridden by setting environment variable ``NUMBA_CUDA_EXCEPTIONS=1``.)
    :param fastmath: When True, enables fastmath optimizations as outlined in
       the :ref:`CUDA Fast Math documentation <cuda-fast-math>`.
    :param max_registers: Request that the kernel is limited to using at most
       this number of registers per thread. The limit may not be respected if
       the ABI requires a greater number of registers than that requested.
       Useful for increasing occupancy.
    :param opt: Whether to compile from LLVM IR to PTX with optimization
                enabled. When ``True``, ``-opt=3`` is passed to NVVM. When
                ``False``, ``-opt=0`` is passed to NVVM. Defaults to ``True``.
    :type opt: bool
    :param lineinfo: If True, generate a line mapping between source code and
       assembly code. This enables inspection of the source code in NVIDIA
       profiling tools and correlation with program counter sampling.
    :type lineinfo: bool
    :param cache: If True, enables the file-based cache for this function.
    :type cache: bool
    :param extensions: An optional list of kernel launch parameter handling
                       extensions
    :type extensions: list
    """

    if config.CUDA_DEBUGINFO_DEFAULT and 'debug' not in options:
        options['debug'] = True

    if config.CUDA_EXCEPTIONS_DEFAULT and 'exceptions' not in options:
        options['exceptions'] = True

    link = options.get('link')
    debug = options.get('debug')
    opt = options.get('opt', True)
    lineinfo = options.get('lineinfo')
    fastmath = options.get('fastmath', False)

    # Checks for erroneous options and combinations of options

    if link and config.ENABLE_CUDASIM:
        raise NotImplementedError('Cannot link PTX in the simulator')

    if options.get('boundscheck'):
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if options.get('argtypes') is not None:
        msg = _msg_deprecated_signature_arg.format('argtypes')
        raise DeprecationError(msg)
    if options.get('restype') is not None:
        msg = _msg_deprecated_signature_arg.format('restype')
        raise DeprecationError(msg)
    if options.get('bind') is not None:
        msg = _msg_deprecated_signature_arg.format('bind')
        raise DeprecationError(msg)

    if device and link:
        raise ValueError("link keyword invalid for device function")

    # Checks for conditions that warrant a warning

    if debug and opt:
        msg = ("debug=True with opt=True (the default) "
               "is not supported by CUDA. This may result in a crash"
               " - set debug=False or opt=False.")
        warn(NumbaInvalidConfigWarning(msg))

    if debug and lineinfo:
        msg = ("debug and lineinfo are mutually exclusive. Use debug to get "
               "full debug info (this disables some optimizations), or "
               "lineinfo for line info only with code generation unaffected.")
        warn(NumbaInvalidConfigWarning(msg))

    if sigutils.is_signature(func_or_sig):
        signatures = [func_or_sig]
        specialized = True
    elif isinstance(func_or_sig, list):
        signatures = func_or_sig
        specialized = False
    else:
        signatures = None

    if signatures is not None:
        if config.ENABLE_CUDASIM:
            def jitwrapper(func):
                return FakeCUDAKernel(func, device=device, fastmath=fastmath)
            return jitwrapper

        def _jit(func):
            disp = CUDADispatcher(func, targetoptions=options, device=device,
                                  extensions=extensions)

            if cache:
                disp.enable_caching()

            for sig in signatures:
                if device:
                    from numba.core import typeinfer
                    with typeinfer.register_dispatcher(disp):
                        disp.compile(sig)
                else:
                    disp.compile_kernel(sig)

            disp._specialized = specialized
            disp.disable_compile()

            return disp

        return _jit
    else:
        if func_or_sig is None:
            if config.ENABLE_CUDASIM:
                def autojitwrapper(func):
                    return FakeCUDAKernel(func, device=device,
                                          fastmath=fastmath)
            else:
                def autojitwrapper(func):
                    return jit(func, device=device, cache=cache,
                               extensions=extensions, **options)

            return autojitwrapper
        # func_or_sig is a function
        else:
            if config.ENABLE_CUDASIM:
                return FakeCUDAKernel(func_or_sig, device=device,
                                      fastmath=fastmath)
            else:
                disp = CUDADispatcher(func_or_sig, targetoptions=options,
                                      device=device, extensions=extensions)

                if cache:
                    disp.enable_caching()

                return disp


def declare_device(name, sig):
    """
    Declare the signature of a foreign function. Returns a descriptor that can
    be used to call the function from a Python kernel.

    :param name: The name of the foreign function.
    :type name: str
    :param sig: The Numba signature of the function.
    """
    argtypes, restype = sigutils.normalize_signature(sig)
    if restype is None:
        msg = 'Return type must be provided for device declarations'
        raise TypeError(msg)

    return declare_device_function(name, restype, argtypes)
