from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.tests.cudapy.usecases import CUDAUseCase
from numba.tests.test_numbers import TestViewIntFloatBase


class TestViewIntFloat(TestViewIntFloatBase, CUDATestCase):
    @classmethod
    def setUpClass(cls):
        def jit(self, f):
            inner = cuda.jit(device=True)(f)

            @cuda.jit
            def outer(r, x):
                r[()] = inner(x[()])

            usecase = CUDAUseCase(outer)
            usecase.py_func = f
            return usecase

        cls.jit = jit


if __name__ == '__main__':
    unittest.main()
