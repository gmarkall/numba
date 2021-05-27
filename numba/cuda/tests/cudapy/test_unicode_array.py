import numpy as np
from numba import cuda
from numba.core.utils import pysignature
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba.tests.test_unicode_array import TestUnicodeArray


kernel_template = """\
@cuda.jit
def kernel(__result, {params}):
    __result[0] = devfunc({params})
"""


@unittest.skip('totally busted')
class TestUnicodeArray(CUDATestCase, TestUnicodeArray):
    def _test(self, pyfunc, cfunc, *args, **kwargs):
        if kwargs:
            self.skipTest('kwargs not supported in CUDA unicode test')

        expected = pyfunc(*args)
        print(type(expected), expected)

        if isinstance(expected, (bool, np.bool_)):
            result = np.zeros(1, dtype=np.bool_)
        elif isinstance(expected, str):
            dtype = f"<U{len(expected)}"
            result = np.zeros(1, dtype=dtype)
        else:
            self.skipTest('Data type not supported yet')

        sig = pysignature(pyfunc)
        n_params = len(sig.parameters)

        if n_params > 4:
            self.skipTest(f'{n_params} parameters not supported yet')

        devfunc = cuda.jit(device=True)(pyfunc)

        if n_params == 1:
            def kernel(result, p0):
                result[0] = devfunc(p0)
        elif n_params == 2:
            def kernel(result, p0, p1):
                result[0] = devfunc(p0, p1)
        elif n_params == 3:
            def kernel(result, p0, p1, p2):
                result[0] = devfunc(p0, p1, p2)
        elif n_params == 4:
            def kernel(result, p0, p1, p2, p3):
                result[0] = devfunc(p0, p1, p2, p3)

        kernel = cuda.jit(kernel)
        kernel[1, 1](result, *args)

        self.assertPreciseEqual(result[0], expected)

    def test_join(self):
        self.skipTest("Long compile")


if __name__ == '__main__':
    unittest.main()
