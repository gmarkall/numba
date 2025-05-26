# See also numba.cuda.tests.test_alignment

import numpy as np
from numba import from_dtype, njit, void
from numba.tests.support import TestCase


class TestAlignment(TestCase):

    def test_record_alignment(self):
        # rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')], align=True)
        # rec = from_dtype(rec_dtype)
        rec = Record([
            ('a', {'type': int32, 'offset': 0, 'alignment': None, 'title': None, }),
            ('b', {'type': float64, 'offset': 8, 'alignment': None, 'title': None, })
        ], 16, True)

        @njit((rec[:],))
        def foo(a):
            for i in range(a.size):
                a[i].a = a[i].b

        a_recarray = np.recarray(3, dtype=rec_dtype)
        for i in range(a_recarray.size):
            a_rec = a_recarray[i]
            a_rec.a = 0
            a_rec.b = (i + 1) * 123

        foo(a_recarray)
        np.testing.assert_equal(a_recarray.a, a_recarray.b)

    def test_record_misaligned(self):
        # rec_dtype = np.dtype([('a', 'int32'), ('b', 'float64')])
        # rec = from_dtype(rec_dtype)
        Record([
            ('a', {'type': int32, 'offset': 0, 'alignment': None, 'title': None, }),
            ('b', {'type': float64, 'offset': 4, 'alignment': None, 'title': None, })
        ], 12, False)

        # Unlike the CUDA target, this will not generate an error
        @njit((rec[:],))
        def foo(a):
            for i in range(a.size):
                a[i].a = a[i].b


if __name__ == '__main__':
    unittest.main()
