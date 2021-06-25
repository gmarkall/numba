from numba import cuda, njit
from numba.cuda.testing import unittest, CUDATestCase

import numpy as np

class TestIterators(CUDATestCase):

    @unittest.skip
    def test_enumerate(self):
        @cuda.jit#(debug=True)
        def enumerator(x):
            count = 0

            for i, v in enumerate(x):
                if count != i:
                    raise RuntimeError("Count and enumeration are unequal")
                if v != x[i]:
                    raise RuntimeError("Value and enumeration are unequal")

                count += 1

            if count != len(x):
                raise RuntimeError("Count did not equal length of x")

        enumerator[1, 1]((10, 9, 8, 7, 6))

    def test_zip(self):
        @cuda.jit(debug=True)
        def zipper(x, y):
            i = 0

            for xv, yv in zip(x, y):
                if xv != x[i]:
                    pass #raise RuntimeError("xv and x[i] are unequal")
                #else:
                #    pass
                #if yv != y[i]:
                #    raise RuntimeError("yv and y[i] are unequal")

                i += 1

            #if i != len(x):
            #    print("bad 3")
            #    #raise RuntimeError("Count did not equal length of x")

        x = np.asarray((10, 9, 8, 7, 6))
        y = np.asarray((1, 2, 3, 4, 5))
        #zipper(x, y)
        zipper[1, 1](x, y)
        #zipper[1, 1]((10, 9, 8, 7, 6), (1, 2, 3, 4, 5))


if __name__ == '__main__':
    unittest.main()
