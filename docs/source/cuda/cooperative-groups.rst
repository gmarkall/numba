.. _cuda-cooperative-groups:

==================
Cooperative Groups
==================

.. warning:: This feature is experimental. The supported features may change
    with or without notice.

Support for a limited set of Cooperative Groups operations on Thread Groups is
provided.

Blog post: https://devblogs.nvidia.com/cooperative-groups/

Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups

.. autoclass:: numba.cuda.cg
   :members: this_thread_block
   :member-order: bysource


.. autoclass:: numba.cuda.ThreadGroup
   :members: size, thread_rank, sync
   :member-order: bysource

.. autoclass:: numba.cuda.ThreadBlock
   :members: tiled_partition
   :member-order: bysource

