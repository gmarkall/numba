import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice


def use_sincos(s, c, x):
    i = cuda.grid(1)

    if i < len(x):
        sr, cr = libdevice.sincos(x[i])
        s[i] = sr
        c[i] = cr


class TestCudaMath(CUDATestCase):
    def test_sincos(self):
        arr = np.arange(100, dtype=np.float64)
        sres = np.zeros_like(arr)
        cres = np.zeros_like(arr)

        cufunc = cuda.jit(use_sincos)
        cufunc[4, 32](sres, cres, arr)

        np.testing.assert_allclose(np.cos(arr), cres)
        np.testing.assert_allclose(np.sin(arr), sres)


if __name__ == '__main__':
    unittest.main()
