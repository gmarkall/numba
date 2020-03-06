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


Asynchronous allocation and deallocation
----------------------------------------

The present EMM Plugin interface does not provide support for asynchronous
allocation and deallocation. This may be added to a future version of the
interface.


Implementing an EMM Plugin
==========================

An EMM Plugin is implemented by deriving from
:class:`~numba.cuda.BaseCUDAMemoryManager`. A summary of considerations for the
implementation follows:

- Numba instantiates one instance of the EMM Plugin class per context. The
  context that owns an EMM Plugin object is accessible through ``self.context``,
  if required.
- The EMM Plugin is transparent to any code that uses Numba - all its methods
  are invoked by Numba, and never need to be called by code that uses Numba.
- The allocation methods ``memalloc``, ``memhostalloc``, and ``mempin``, should
  use the underlying library to allocate and/or pin device or host memory, and
  construct an instance of a :ref:`memory pointer <memory-pointers>`
  representing the memory to return back to Numba. These methods are always
  called when the current CUDA context is the context that owns the EMM Plugin
  instance.
- The ``initialize`` method is called by Numba prior to the first use of the EMM
  Plugin object for a constant. This method should do anything required to
  prepare the underlying library for allocations in the current context. This
  method may be called multiple times, and must not invalidate previous state
  when it is called.
- The ``reset`` method is called when all allocations in the context are to be
  cleaned up. It may be called even prior to ``initialize``, and an EMM Plugin
  implementation needs to guard against this.
- To support inter-GPU communication, the ``get_ipc_handle`` method should
  provide an :class:`~numba.cuda.cudadrv.driver.IpcHandle` for a given
  :class:`~numba.cuda.MemoryPointer` instance. This method is part of the EMM
  interface (rather than being handled within Numba) because the base address of
  the allocation is only known by the underlying library. Closing an IPC handle
  is handled internally within Numba.
- It is optional to implement the ``get_memory_info`` method, which provides a
  count of the total and free memory on the device for the context. It is
  preferrable to implement the method, but this may not be practical for all
  allocators.
- The ``defer_cleanup`` method should return a context manager that ensures that
  expensive cleanup operations are avoided whilst it is active. The nuances of
  this will vary between plugins, so the plugin documentation should include an
  explanation of how deferring cleanup affects deallocations, and performance in
  general.
- The ``interface_version`` property is used to ensure that the plugin version
  matches the interface provided by the version of Numba. At present, this
  should always be 1.

Full documentation for the base class follows

.. autoclass:: numba.cuda.BaseCUDAMemoryManager
   :members: memalloc, memhostalloc, mempin, initialize, get_ipc_handle,
             get_memory_info, reset, defer_cleanup, interface_version
   :member-order: bysource


.. _host-only-cuda-memory-manager:

The Host-Only CUDA Memory Manager
---------------------------------

.. autoclass:: numba.cuda.HostOnlyCUDAMemoryManager


Associated classes and structures
=================================

.. _memory-pointers:

Memory Pointers
---------------

.. autoclass:: numba.cuda.MemoryPointer

The ``AutoFreePointer`` class need not be used directly, but is documented here
as it is subclassed by :class:`numba.cuda.MappedMemory`:

.. autoclass:: numba.cuda.cudadrv.driver.AutoFreePointer

.. autoclass:: numba.cuda.MappedMemory

.. autoclass:: numba.cuda.PinnedMemory


Other structures
----------------

.. autoclass:: numba.cuda.cudadrv.driver.MemoryInfo

.. autoclass:: numba.cuda.cudadrv.driver.IpcHandle


.. _setting-emm-plugin:

Setting the EMM Plugin
======================
