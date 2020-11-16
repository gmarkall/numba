from numba import cuda
from torch.utils import dlpack

import numpy as np


h_x = np.arange(10).astype(np.float64).reshape((2, 5))
x = cuda.to_device(h_x)
#print(x[3, 2])
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
# 0x7fdbf5400000
# On C side:
# data = 0x7fdbf5400000
# ctx = (device_type = 2, device_id = 0)
# dtype = (code = 2, bits = 64, lanes = 1)
# ndim = 2
# shape = (2, 5)
# strides = (5, 1)
# <capsule object "dltensor" at 0x7fdc8d79d480>
# 2 torch.Size([2, 5])
# tensor([[0., 1., 2., 3., 4.],
#         [5., 6., 7., 8., 9.]], device='cuda:0', dtype=torch.float64)
# In destructor
# Capsule already consumed
# In deleter!
