from numba import cuda
import numpy as np

h_x = np.arange(10).astype(np.float64)

x = cuda.to_device(h_x)
dlp = x.to_dlpack()

# Should not result in anything getting added to pending deallocs
del x
# Clear all pending deallocations, just to be sure
ctx = cuda.current_context()
ctx.memory_manager.deallocations.clear()

# If we do this earlier Numba is using RMM, which is less than ideal
# for this prototype
import cudf

cudf_obj = cudf.from_dlpack(dlp)
del dlp

print(type(cudf_obj))
print(cudf_obj)

# The array of 80 bytes (10 doubles) should have been deleted by now,
# or will be deleted by this call
ctx.memory_manager.deallocations.clear()
