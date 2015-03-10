from numba import mviewbuf
import numpy as np
dt = np.dtype([('a', np.int32), ('b', np.float32)])
rec = np.recarray(1, dt)[0]
bp = mviewbuf.BufferProxy(rec)
ptr = mviewbuf.memoryview_get_buffer(bp)
