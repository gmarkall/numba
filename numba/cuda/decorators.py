import inspect
import logging

from numba.core import types, config, sigutils
from numba.core.errors import DeprecationError
from .compiler import (compile_device, declare_device_function, Dispatcher,
                       compile_device_template)
from .simulator.kernel import FakeCUDAKernel


_logger = logging.getLogger(__name__)

_msg_deprecated_signature_arg = ("Deprecated keyword argument `{0}`. "
                                 "Signatures should be passed as the first "
                                 "positional argument.")


def jitdevice(func, link=[], debug=None, inline=False, opt=True):
    """Wrapper for device-jit.
    """
    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug
    if link:
        raise ValueError("link keyword invalid for device function")
    return compile_device_template(func, debug=debug, inline=inline, opt=opt)


def jit(func_or_sig=None, device=False, inline=False, link=[], debug=None,
        opt=True, **kws):
    """
    JIT compile a python function conforming to the CUDA Python specification.
    If a signature is supplied, then a function is returned that takes a
    function to compile.

    :param func_or_sig: A function to JIT compile, or a signature of a function
       to compile. If a function is supplied, then a
       :class:`numba.cuda.compiler.AutoJitCUDAKernel` is returned. If a
       signature is supplied, then a function is returned. The returned
       function accepts another function, which it will compile and then return
       a :class:`numba.cuda.compiler.AutoJitCUDAKernel`.

       .. note:: A kernel cannot have any return value.
    :param device: Indicates whether this is a device function.
    :type device: bool
    :param link: A list of files containing PTX source to link with the function
    :type link: list
    :param debug: If True, check for exceptions thrown when executing the
       kernel. Since this degrades performance, this should only be used for
       debugging purposes.  Defaults to False.  (The default value can be
       overridden by setting environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
    :param fastmath: If true, enables flush-to-zero and fused-multiply-add,
       disables precise division and square root. This parameter has no effect
       on device function, whose fastmath setting depends on the kernel function
       from which they are called.
    :param max_registers: Limit the kernel to using at most this number of
       registers per thread. Useful for increasing occupancy.
    :param opt: Whether to compile from LLVM IR to PTX with optimization
                enabled. When ``True``, ``-opt=3`` is passed to NVVM. When
                ``False``, ``-opt=0`` is passed to NVVM. Defaults to ``True``.
    :type opt: bool
    """
    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug

    if link and config.ENABLE_CUDASIM:
        raise NotImplementedError('Cannot link PTX in the simulator')

    if kws.get('boundscheck'):
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if kws.get('argtypes') is not None:
        msg = _msg_deprecated_signature_arg.format('argtypes')
        raise DeprecationError(msg)
    if kws.get('restype') is not None:
        msg = _msg_deprecated_signature_arg.format('restype')
        raise DeprecationError(msg)
    if kws.get('bind') is not None:
        msg = _msg_deprecated_signature_arg.format('bind')
        raise DeprecationError(msg)

    fastmath = kws.get('fastmath', False)
    if not sigutils.is_signature(func_or_sig):
        if func_or_sig is None:
            if config.ENABLE_CUDASIM:
                def autojitwrapper(func):
                    return FakeCUDAKernel(func, device=device,
                                          fastmath=fastmath, debug=debug)
            else:
                def autojitwrapper(func):
                    return jit(func, device=device, debug=debug, opt=opt, **kws)

            return autojitwrapper
        # func_or_sig is a function
        else:
            if config.ENABLE_CUDASIM:
                return FakeCUDAKernel(func_or_sig, device=device,
                                      fastmath=fastmath, debug=debug)
            elif device:
                return jitdevice(func_or_sig, debug=debug, opt=opt, **kws)
            else:
                targetoptions = kws.copy()
                targetoptions['debug'] = debug
                targetoptions['opt'] = opt
                targetoptions['link'] = link
                sigs = None
                return Dispatcher(func_or_sig, sigs,
                                  targetoptions=targetoptions)

    else:
        if config.ENABLE_CUDASIM:
            def jitwrapper(func):
                return FakeCUDAKernel(func, device=device, fastmath=fastmath,
                                      debug=debug)
            return jitwrapper

        if isinstance(func_or_sig, list):
            msg = 'Lists of signatures are not yet supported in CUDA'
            raise ValueError(msg)
        elif sigutils.is_signature(func_or_sig):
            sigs = [func_or_sig]
        else:
            raise ValueError("Expecting signature or list of signatures")

        for sig in sigs:
            argtypes, restype = sigutils.normalize_signature(sig)

            if restype and not device and restype != types.void:
                raise TypeError("CUDA kernel must have void return type.")

        def kernel_jit(func):
            targetoptions = kws.copy()
            targetoptions['debug'] = debug
            targetoptions['link'] = link
            targetoptions['opt'] = opt
            return Dispatcher(func, sigs, targetoptions=targetoptions)

        def device_jit(func):
            return compile_device(func, restype, argtypes, inline=inline,
                                  debug=debug)

        if device:
            return device_jit
        else:
            return kernel_jit


def declare_device(name, sig):
    argtypes, restype = sigutils.normalize_signature(sig)
    return declare_device_function(name, restype, argtypes)


def jit_module(**kwargs):
    """ Automatically ``jit``-wraps functions defined in a Python module. By
    default, wrapped functions are treated as device functions rather than
    kernels - pass ``device=False`` to treat functions as kernels.

    Note that ``jit_module`` should be called following the declaration of all
    functions to be jitted. This function may be called multiple times within
    a module with different options, and any new function declarations since
    the previous ``jit_module`` call will be wrapped with the options provided
    to the current call to ``jit_module``.

    Note that only functions which are defined in the module ``jit_module`` is
    called from are considered for automatic jit-wrapping.  See the Numba
    documentation for more information about what can/cannot be jitted.

    :param kwargs: Keyword arguments to pass to ``jit`` such as ``device``
                   or ``opt``.

    """
    if 'device' not in kwargs:
        kwargs['device'] = True

    # Get the module jit_module is being called from
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    # Replace functions in module with jit-wrapped versions
    for name, obj in module.__dict__.items():
        if inspect.isfunction(obj) and inspect.getmodule(obj) == module:
            _logger.debug("Auto decorating function {} from module {} with jit "
                          "and options: {}".format(obj, module.__name__,
                                                   kwargs))
            module.__dict__[name] = jit(obj, **kwargs)
