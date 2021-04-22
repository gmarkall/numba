# We import * from simulator here because * is imported from simulator_init by
# numba.cuda.__init__.
from .simulator import *  # noqa: F403, F401

from numba.core import sigutils
from .simulator.kernel import FakeCUDAKernel


def jit(func_or_sig=None, device=False, inline=False, link=[], debug=None,
        opt=True, fastmath=False, **kws):
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
    """

    if link:
        raise NotImplementedError('Cannot link PTX in the simulator')

    if kws.get('boundscheck'):
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if func_or_sig is None or sigutils.is_signature(func_or_sig):
        def jitwrapper(func):
            return FakeCUDAKernel(func, device=device, fastmath=fastmath)
        return jitwrapper
    else:
        return FakeCUDAKernel(func_or_sig, device=device, fastmath=fastmath)


def is_available():
    """Returns a boolean to indicate the availability of a CUDA GPU.
    """
    # Simulator is always available
    return True


def cuda_error():
    """Returns None or an exception if the CUDA driver fails to initialize.
    """
    # Simulator never fails to initialize
    return None
