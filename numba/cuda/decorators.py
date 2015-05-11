from __future__ import print_function, absolute_import, division
from numba import config, sigutils, types
from warnings import warn
from .compiler import (compile_kernel, compile_device, declare_device_function,
                       AutoJitCUDAKernel, compile_device_template)
from .simulator.kernel import FakeCUDAKernel

def jitdevice(func, link=[], debug=False, inline=False):
    """Wrapper for device-jit.
    """
    if link:
        raise ValueError("link keyword invalid for device function")
    return compile_device_template(func, debug=debug, inline=inline)


def jit(restype=None, argtypes=None, device=False, inline=False, bind=True,
        link=[], debug=False, **kws):
    """JIT compile a python function conforming to
    the CUDA-Python specification.

    To define a CUDA kernel that takes two int 1D-arrays::

        @cuda.jit('void(int32[:], int32[:])')
        def foo(aryA, aryB):
            ...

    .. note:: A kernel cannot have any return value.

    To launch the cuda kernel::

        griddim = 1, 2
        blockdim = 3, 4
        foo[griddim, blockdim](aryA, aryB)


    ``griddim`` is the number of thread-block per grid.
    It can be:

    * an int;
    * tuple-1 of ints;
    * tuple-2 of ints.

    ``blockdim`` is the number of threads per block.
    It can be:

    * an int;
    * tuple-1 of ints;
    * tuple-2 of ints;
    * tuple-3 of ints.

    The above code is equaivalent to the following CUDA-C.

    .. code-block:: c

        dim3 griddim(1, 2);
        dim3 blockdim(3, 4);
        foo<<<griddim, blockdim>>>(aryA, aryB);


    To access the compiled PTX code::

        print foo.ptx


    To define a CUDA device function that takes two ints and returns a int::

        @cuda.jit('int32(int32, int32)', device=True)
        def bar(a, b):
            ...

    To force inline the device function::

        @cuda.jit('int32(int32, int32)', device=True, inline=True)
        def bar_forced_inline(a, b):
            ...

    A device function can only be used inside another kernel.
    It cannot be called from the host.

    Using ``bar`` in a CUDA kernel::

        @cuda.jit('void(int32[:], int32[:], int32[:])')
        def use_bar(aryA, aryB, aryOut):
            i = cuda.grid(1) # global position of the thread for a 1D grid.
            aryOut[i] = bar(aryA[i], aryB[i])

    When the function signature is not given, this decorator behaves like
    autojit.


    The following addition options are available for kernel functions only.
    They are ignored in device function.

    - fastmath: bool
        Enables flush-to-zero for denormal float;
        Enables fused-multiply-add;
        Disables precise division;
        Disables precise square root.
    """

    if link and config.ENABLE_CUDASIM:
        raise NotImplementedError('Cannot link PTX in the simulator')

    if argtypes is None and not sigutils.is_signature(restype):
        if restype is None:
            if config.ENABLE_CUDASIM:
                def autojitwrapper(func):
                    return FakeCUDAKernel(func, device=device, fastmath=fastmath,
                                          debug=debug)
            else:
                def autojitwrapper(func):
                    return jit(func, device=device, bind=bind, **kws)

            return autojitwrapper
        # restype is a function
        else:
            if config.ENABLE_CUDASIM:
                return FakeCUDAKernel(restype, device=device, fastmath=fastmath,
                                       debug=debug)
            elif device:
                return jitdevice(restype, **kws)
            else:
                targetoptions = kws.copy()
                targetoptions['debug'] = debug
                return AutoJitCUDAKernel(restype, bind=bind, targetoptions=targetoptions)
            #return decor(restype)

    else:
        fastmath = kws.get('fastmath', False)
        if config.ENABLE_CUDASIM:
            def jitwrapper(func):
                return FakeCUDAKernel(func, device=device, fastmath=fastmath,
                                      debug=debug)
            return jitwrapper

        restype, argtypes = convert_types(restype, argtypes)

        if restype and not device and restype != types.void:
            raise TypeError("CUDA kernel must have void return type.")

        def kernel_jit(func):
            kernel = compile_kernel(func, argtypes, link=link, debug=debug,
                                    inline=inline, fastmath=fastmath)

            # Force compilation for the current context
            if bind:
                kernel.bind()

            return kernel

        def device_jit(func):
            return compile_device(func, restype, argtypes, inline=inline,
                                  debug=debug)

        if device:
            return device_jit
        else:
            return kernel_jit


def autojit(*args, **kwargs):
    warn('autojit is deprecated and will be removed in a future release. Use jit instead.')
    return jit(*args, **kwargs)


def declare_device(name, restype=None, argtypes=None):
    restype, argtypes = convert_types(restype, argtypes)
    return declare_device_function(name, restype, argtypes)


def convert_types(restype, argtypes):
    # eval type string
    if sigutils.is_signature(restype):
        assert argtypes is None
        argtypes, restype = sigutils.normalize_signature(restype)

    return restype, argtypes

