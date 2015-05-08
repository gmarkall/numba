CUDA Host API
=============

Device Management
-----------------

.. autofunction:: numba.cuda.is_available

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

.. autofunction:: numba.cuda.synchronize

.. autofunction:: numba.cuda.select_device
.. autofunction:: numba.cuda.get_current_device
.. autofunction:: numba.cuda.list_devices
.. autofunction:: numba.cuda.detect

Profiling and Measurement
-------------------------

.. autofunction:: numba.cuda.profile_start
.. autofunction:: numba.cuda.profile_stop
.. autofunction:: numba.cuda.profiling

.. autofunction:: numba.cuda.event
.. autofunction:: numba.cuda.event_elapsed_time

.. autoclass:: numba.cuda.cudadrv.driver.Event
   :members: query, record, synchronize, wait

Stream Management
-----------------

.. autofunction:: numba.cuda.stream

.. autoclass:: numba.cuda.cudadrv.driver.Stream
   :members: synchronize, auto_synchronize

