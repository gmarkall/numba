import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.usecases import CPUUseCase, CUDAUseCase

import unittest


def reinterpret_array_type(byte_arr, start, stop, output):
    # Tested with just one thread
    val = byte_arr[start:stop].view(np.int32)[0]
    output[0] = val


class TestCudaArrayMethods(CUDATestCase):
    def test_reinterpret_array_type(self):
        """
        Reinterpret byte array as int32 in the GPU.
        """
        pyfunc = reinterpret_array_type
        kernel = cuda.jit(pyfunc)

        byte_arr = np.arange(256, dtype=np.uint8)
        itemsize = np.dtype(np.int32).itemsize
        for start in range(0, 256, itemsize):
            stop = start + itemsize
            expect = byte_arr[start:stop].view(np.int32)[0]

            output = np.zeros(1, dtype=np.int32)
            kernel[1, 1](byte_arr, start, stop, output)

            got = output[0]
            self.assertEqual(expect, got)

    # Modified from numba.tests.test_array_methods

    def _test_method(self, arr_func, np_func):

        all_dtypes = (np.float64, np.float32, np.int64, np.int32,
                      np.complex64, np.complex128, np.uint32, np.uint64)
        # np.timedelta64)
        all_test_arrays = [
            (np.ones((7, 6, 5, 4, 3), arr_dtype),
             np.ones(1, arr_dtype),
             np.ones((7, 3), arr_dtype) * -5)
            for arr_dtype in all_dtypes]

        array_pyfunc = CPUUseCase(arr_func)
        array_cfunc = CUDAUseCase(cuda.jit(arr_func))

        np_pyfunc = CPUUseCase(np_func)
        np_cfunc = CUDAUseCase(cuda.jit(np_func))
        for arr_list in all_test_arrays:
            for arr in arr_list:
                with self.subTest(method="array method", dtype=arr.dtype):
                    self.assertPreciseEqual(array_pyfunc(arr), array_cfunc(arr))
                with self.subTest(method="module method", dtype=arr.dtype):
                    self.assertPreciseEqual(np_pyfunc(arr), np_cfunc(arr))

    @unittest.skip
    def test_min(self):
        def array_min(r, x):
            r[()] = x.min()

        def np_min(r, x):
            r[()] = np.min(x)

        self._test_method(array_min, np_min)

    @unittest.skip
    def test_max(self):
        def array_max(r, x):
            r[()] = x.max()

        def np_max(r, x):
            r[()] = np.max(x)

        self._test_method(array_max, np_max)

    def test_sum(self):
        def array_sum(r, x):
            r[()] = x.sum()

        def np_sum(r, x):
            r[()] = np.sum(x)

        self._test_method(array_sum, np_sum)

    def test_prod(self):
        def array_prod(r, x):
            r[()] = x.prod()

        def np_prod(r, x):
            r[()] = np.prod(x)

        self._test_method(array_prod, np_prod)

    def test_mean(self):
        def array_mean(r, x):
            r[()] = x.mean()

        def np_mean(r, x):
            r[()] = np.mean(x)

        self._test_method(array_mean, np_mean)

    @unittest.skip
    def test_var(self):
        def array_var(r, x):
            r[()] = x.var()

        def np_var(r, x):
            r[()] = np.var(x)

        self._test_method(array_var, np_var)

    @unittest.skip
    def test_std(self):
        def array_std(r, x):
            r[()] = x.std()

        def np_std(r, x):
            r[()] = np.std(x)

        self._test_method(array_std, np_std)


if __name__ == '__main__':
    unittest.main()
