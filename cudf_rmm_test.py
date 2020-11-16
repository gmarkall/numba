from numba import cuda
import numpy as np
import cudf

h_x = np.arange(10).astype(np.float64)

x = cuda.to_device(h_x)
dlp = x.to_dlpack()

# Should not result in anything getting added to pending deallocs
del x
# Clear all pending deallocations, just to be sure
ctx = cuda.current_context()
ctx.memory_manager.deallocations.clear()

cudf_obj = cudf.from_dlpack(dlp)
del dlp

print(type(cudf_obj))
print(cudf_obj)

# The array of 80 bytes (10 doubles) should have been deleted by now,
# or will be deleted by this call
ctx.memory_manager.deallocations.clear()

# Output:
#
# On C side:
# data = 0x7f4f4be00000
# ctx = (device_type = 2, device_id = 0)
# dtype = (code = 2, bits = 64, lanes = 1)
# ndim = 1
# shape = (10)
# strides = (1)
# /home/gmarkall/miniconda3/envs/cudf-0.16/lib/python3.8/site-packages/cudf/io/dlpack.py:33: UserWarning: WARNING: cuDF from_dlpack() assumes column-major (Fortran order) input. If the input tensor is row-major, transpose it before passing it to this function.
#   res = libdlpack.from_dlpack(pycapsule_obj)
# In deleter!
# In destructor
# Capsule already consumed
# <class 'cudf.core.series.Series'>
# 0    0.0
# 1    1.0
# 2    2.0
# 3    3.0
# 4    4.0
# 5    5.0
# 6    6.0
# 7    7.0
# 8    8.0
# 9    9.0
# dtype: float64
