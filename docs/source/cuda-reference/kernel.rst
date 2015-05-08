CUDA Kernel API
===============

.. autofunction:: numba.cuda.jit
.. autofunction:: numba.cuda.autojit
.. autofunction:: numba.cuda.declare_device

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

.. autofunction:: numba.cuda.shared
.. autofunction:: numba.cuda.local
.. autofunction:: numba.cuda.const

Synchronization and Atomic Operations
-------------------------------------

.. autoclass:: numba.cuda.atomic
   :members: add, max
.. autofunction:: numba.cuda.syncthreads
