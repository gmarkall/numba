from numba import cuda
from torch.utils import dlpack

import numpy as np


h_x = np.arange(10).astype(np.float64)
x = cuda.to_device(h_x)
print(x[3])
print("0x%x" % x.gpu_data._mem.handle.value)

dlp = x.to_dlpack()

print(dlp)

t_x = dlpack.from_dlpack(dlp)
print(t_x.ndim, t_x.shape)

print(t_x)

del dlp

del x

# Output:
#
# 3.0
# 0x7f0e63400000
# On C side:
# data = 0x7f0e63400000
# ctx = (device_type = 2, device_id = 0)
# dtype = (code = 2, bits = 64, lanes = 1)
# ndim = 1
# shape = (10)
# strides = (1)
# <capsule object "dltensor" at 0x7f0ef9414480>
# 1 torch.Size([10])
# tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda:0',
#        dtype=torch.float64)
# In destructor
# Capsule already consumed
# In deleter!
