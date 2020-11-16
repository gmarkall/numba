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
