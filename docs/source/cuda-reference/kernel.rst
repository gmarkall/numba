CUDA Kernel API
===============

Kernel declaration
------------------

The ``@cuda.jit`` decorator is used to create a CUDA kernel:

.. autofunction:: numba.cuda.jit

.. autoclass:: numba.cuda.compiler.AutoJitCUDAKernel
   :members: inspect_asm, inspect_llvm, inspect_types, specialize

Individual specialized kernels are instances of
:class:`numba.cuda.compiler.CUDAKernel`:

.. autoclass:: numba.cuda.compiler.CUDAKernel
   :members: bind, ptx, device, inspect_llvm, inspect_asm, inspect_types

Thread Indexing
---------------

.. autofunction:: numba.cuda.threadIdx
.. autofunction:: numba.cuda.blockIdx
.. autofunction:: numba.cuda.blockDim
.. autofunction:: numba.cuda.gridDim
.. autoattribute:: numba.cuda.grid
.. autoattribute:: numba.cuda.gridsize

Memory Management
-----------------

.. autoclass:: numba.cuda.shared
   :members: array
.. autofunction:: numba.cuda.local
.. autofunction:: numba.cuda.const

Synchronization and Atomic Operations
-------------------------------------

.. autoclass:: numba.cuda.atomic
   :members: add, max
.. autofunction:: numba.cuda.syncthreads
