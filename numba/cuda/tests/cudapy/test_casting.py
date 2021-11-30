import numpy as np

from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import unittest


def to_int8(x):
    return np.int8(x)


def to_int16(x):
    return np.int16(x)


def to_int32(x):
    return np.int32(x)


def to_int64(x):
    return np.int64(x)


def to_uint8(x):
    return np.uint8(x)


def to_uint16(x):
    return np.uint16(x)


def to_uint32(x):
    return types.uint32(x)


def to_uint64(x):
    return types.uint64(x)


def to_float16(x):
    # When division and operators on float16 types are supported, this should
    # be changed to match the implementation in to_float32.
    return cuda.fp16.hmul(np.float16(x), 2)


def to_float32(x):
    return np.float32(x) / np.float32(2)


def to_float64(x):
    return np.float64(x) / np.float64(2)


def to_complex64(x):
    return np.complex64(x)


def to_complex128(x):
    return np.complex128(x)


class TestCasting(CUDATestCase):
    def _create_wrapped(self, pyfunc, intype, outtype):
        wrapped_func = cuda.jit(device=True)(pyfunc)

        @cuda.jit
        def cuda_wrapper_fn(arg, res):
            res[0] = wrapped_func(arg[0])

        def wrapper_fn(arg):
            argarray = np.zeros(1, dtype=intype)
            argarray[0] = arg
            resarray = np.zeros(1, dtype=outtype)
            cuda_wrapper_fn[1, 1](argarray, resarray)
            return resarray[0]

        return wrapper_fn

    def test_float_to_int(self):
        pyfuncs = (to_int8, to_int16, to_int32, to_int64)
        totys = (np.int8, np.int16, np.int32, np.int64)
        fromtys = (np.float16, np.float32, np.float64)

        for pyfunc, toty in zip(pyfuncs, totys):
            for fromty in fromtys:
                with self.subTest(fromty=fromty, toty=toty):
                    cfunc = self._create_wrapped(pyfunc, fromty, toty)
                    self.assertEqual(cfunc(12.3), pyfunc(12.3))
                    self.assertEqual(cfunc(12.3), int(12.3))
                    self.assertEqual(cfunc(-12.3), pyfunc(-12.3))
                    self.assertEqual(cfunc(-12.3), int(-12.3))

    def test_float_to_uint(self):
        pyfuncs = (to_int8, to_int16, to_int32, to_int64)
        totys = (np.uint8, np.uint16, np.uint32, np.uint64)
        fromtys = (np.float16, np.float32, np.float64)

        for pyfunc, toty in zip(pyfuncs, totys):
            for fromty in fromtys:
                with self.subTest(fromty=fromty, toty=toty):
                    cfunc = self._create_wrapped(pyfunc, fromty, toty)
                    self.assertEqual(cfunc(12.3), pyfunc(12.3))
                    self.assertEqual(cfunc(12.3), int(12.3))

    def test_int_to_float(self):
        pyfuncs = (to_float16, to_float32, to_float64)
        totys = (np.float16, np.float32, np.float64)

        for pyfunc, toty in zip(pyfuncs, totys):
            with self.subTest(toty=toty):
                cfunc = self._create_wrapped(pyfunc, np.int64, toty)
                self.assertEqual(cfunc(321), pyfunc(321))

    def test_float_to_float(self):
        pyfuncs = (to_float16, to_float32, to_float64)
        tys = (np.float16, np.float32, np.float64)

        for (pyfunc, fromty), toty in itertools.product(zip(pyfuncs, tys), tys):
            with self.subTest(fromty=fromty, toty=toty):
                cfunc = self._create_wrapped(pyfunc, fromty, toty)
                # For this test we cannot use the pyfunc for comparison because
                # the CUDA target doesn't yet implement division (or operators)
                # for float16 values, so we test by comparing with the computed
                # expression instead.
                np.testing.assert_allclose(cfunc(12.3),
                                           toty(12.3) / toty(2))
                np.testing.assert_allclose(cfunc(-12.3),
                                           toty(-12.3) / toty(2))

    def test_float_to_complex(self):
        pyfuncs = (to_complex64, to_complex128)
        totys = (np.complex64, np.complex128)
        fromtys = (np.float16, np.float32, np.float64)

        for pyfunc, toty in zip(pyfuncs, totys):
            for fromty in fromtys:
                with self.subTest(fromty=fromty, toty=toty):
                    cfunc = self._create_wrapped(pyfunc, fromty, toty)
                    # Here we need to explicitly cast the input to the pyfunc
                    # to match the casting that is automatically applied when
                    # passing the input to the cfunc as part of wrapping it in
                    # an array of type fromtype.
                    np.testing.assert_allclose(cfunc(3.21),
                                               pyfunc(fromty(3.21)))
                    np.testing.assert_allclose(cfunc(-3.21),
                                               pyfunc(fromty(-3.21)) + 0j)


if __name__ == '__main__':
    unittest.main()
