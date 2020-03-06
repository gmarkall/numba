.. _cuda-emm-plugin:

===========================================
External Memory Management plugin interface
===========================================

By default, Numba allocates memory on CUDA devices by interacting with the CUDA
driver API to call functions such as ``cuMemAlloc`` and ``cuMemFree``, which is
suitable for many use cases. When Numba is used in conjunction with other
CUDA-aware libraries, it may be preferable for all memory to be managed by a
single one of these libraries. which may not be Numba.  For example, both CuPy
and the RAPIDS Memory Manager provide pool allocators.  The EMM Plugin interface
facilitates this, by enabling Numba to use another CUDA-aware library for memory
management.

To enable Numba to use another library for memory management, an EMM Plugin
needs to be implemented - this could be a part of the library that provides
memory management, or can be developed externally from it.


Overview of External Memory Management
======================================

When an EMM Plugin is in use (see :ref:`setting-emm-plugin`), Numba will make
all memory allocation using the Plugin. It will never directly call functions
such as ``cuMemAlloc``, ``cuMemFree``, etc. EMM Plugins always take
responsibility for the management of device memory. However, not all CUDA-aware
libraries also support managing host memory, so a facility for Numba to continue
the management of host memory whilst ceding control of device memory to the EMM
will be provided (see :ref:`host-only-cuda-memory-manager`).


Effects on Deallocation Strategies
----------------------------------

Numba's internal :ref:`deallocation-behavior` is designed to increase efficiency
by deferring deallocations until a significant quantity are pending. It also
provides a mechanism for preventing deallocations entirely during critical
sections, using the :func:`~numba.cuda.defer_cleanup` context manager.

When an EMM Plugin is in use, the deallocation strategy is implemented by the
EMM, and Numba's internal deallocation mechanism is not used. The EMM
Plugin could implement:
  
- A similar strategy to the Numba deallocation behaviour, or
- Something more appropriate to the plugin - for example, deallocated memory
  might immediately be returned to a memory pool.

The ``defer_cleanup`` context manager may behave differently with an EMM Plugin
- an EMM Plugin should be accompanied by documentation of the behaviour of the
``defer_cleanup`` context manager when it is in use. For example, a pool
allocator could always immediately return memory to a pool even when the
context manager is in use, but could choose not to free empty pools until
``defer_cleanup`` is not in use.


Management of other objects
---------------------------

In addition to memory, Numba manages the allocation and deallocation of
:ref:`events`, :ref:`streams <streams>`, and modules (modules are a compiled objects,
which is generated for CUDA kernels). The management of streams, events, and
modules is unchanged by the use of an EMM Plugin.



Plugin interfaces
=================

.. autoclass:: numba.cuda.BaseCUDAMemoryManager
   :members: __init__, memalloc, memhostalloc, mempin, initialize,
             get_ipc_handle, get_memory_info, reset, defer_cleanup,
             interface_version

.. autoclass:: numba.cuda.cudadrv.driver.MemoryInfo


.. _host-only-cuda-memory-manager:

The Host-Only CUDA Memory Manager
---------------------------------

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


.. _setting-emm-plugin:

Setting the EMM Plugin
======================
