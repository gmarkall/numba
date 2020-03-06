===========================================
External Memory Management plugin interface
===========================================

.. _cuda-emm-plugin:

By default, Numba allocates memory on CUDA devices by interacting with the CUDA
driver API to call functions such as ``cuMemAlloc`` and ``cuMemFree``. This is
suitable for many use cases. When Numba is used in conjunction with other
CUDA-aware libraries that also allocate memory,


Plugin interfaces
=================

.. autoclass:: numba.cuda.BaseCUDAMemoryManager
   :members: __init__, memalloc, memhostalloc, mempin, initialize,
             get_ipc_handle, get_memory_info, reset, defer_cleanup,
             interface_version

.. autoclass:: numba.cuda.cudadrv.driver.MemoryInfo

.. autoclass:: numba.cuda.HostOnlyCUDAMemoryManager


Memory pointers
===============

.. autoclass:: numba.cuda.MemoryPointer

The ``AutoFreePointer`` class need not be used directly, but is documented here
as it is subclassed by :class:`numba.cuda.MappedMemory`:

.. autoclass:: numba.cuda.cudadrv.driver.AutoFreePointer

.. autoclass:: numba.cuda.MappedMemory

.. autoclass:: numba.cuda.PinnedMemory

IPC
===

.. autoclass:: numba.cuda.cudadrv.driver.IpcHandle
