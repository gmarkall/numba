CUDA Host API
=============

Device Management
-----------------

.. autofunction:: numba.cuda.current_context
.. autofunction:: numba.cuda.require_context

.. autoclass:: numba.cuda.cudadrv.driver.Context
   :members: reset, get_memory_info, push, pop

.. attribute:: numba.cuda.gpus

   A :class:`_DeviceList` instance.

.. autoclass:: numba.cuda.cudadrv.devices._DeviceList
   :members: current
.. autoclass:: numba.cuda.cudadrv.devices._DeviceContextManager

.. autofunction:: numba.cuda.close

Profiling and Measurement
-------------------------
