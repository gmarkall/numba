from __future__ import print_function, division, absolute_import

import sys

import numpy as np
from numba import cuda, numpy_support, types
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba.utils import IS_PY3


def set_a(ary, i, v):
    ary[i].a = v


def set_b(ary, i, v):
    ary[i].b = v


def set_c(ary, i, v):
    ary[i].c = v


def set_record(ary, i, j):
    ary[i] = ary[j]


def record_write_array(ary):
    ary.g = 2
    ary.h[0] = 3.0
    ary.h[1] = 4.0

recordtype = np.dtype(
    [
        ('a', np.float64),
        ('b', np.int32),
        ('c', np.complex64),
        ('d', (np.str, 5))
    ],
    align=True
)

recordtype2 = np.dtype(
    [
        ('e', np.int32),
        ('f', np.float64)
    ],
    align=True
)

recordwitharray = np.dtype(
    [
        ('g', np.int32),
        ('h', np.float32, 2)
    ],
    align=True
)

class TestRecordDtype(unittest.TestCase):

    def _createSampleArrays(self):
        '''
        Set up the data structures to be used with the Numpy and Numba
        versions of functions.

        In this case, both accept recarrays.
        '''
        self.refsample1d = np.recarray(3, dtype=recordtype)
        self.refsample1d2 = np.recarray(3, dtype=recordtype2)
        self.refsample1d3 = np.recarray(3, dtype=recordtype)
        self.refsample1d4 = np.recarray(3, dtype=recordwitharray)

        self.nbsample1d = np.recarray(3, dtype=recordtype)
        self.nbsample1d2 = np.recarray(3, dtype=recordtype2)
        self.nbsample1d3 = np.recarray(3, dtype=recordtype)
        self.nbsample1d4 = np.recarray(3, dtype=recordwitharray)

    def setUp(self):

        self._createSampleArrays()

        for ary in (self.refsample1d, self.nbsample1d):
            for i in range(ary.size):
                x = i + 1
                ary[i]['a'] = x / 2
                ary[i]['b'] = x
                ary[i]['c'] = x * 1j
                ary[i]['d'] = "%d" % x

        for ary2 in (self.refsample1d2, self.nbsample1d2):
            for i in range(ary2.size):
                x = i + 5
                ary2[i]['e'] = x
                ary2[i]['f'] = x / 2

        for ary3 in (self.refsample1d3, self.nbsample1d3):
            for i in range(ary3.size):
                x = i + 10
                ary3[i]['a'] = x / 2
                ary3[i]['b'] = x
                ary3[i]['c'] = x * 1j
                ary3[i]['d'] = "%d" % x


    def get_cfunc(self, pyfunc, argspec):
        return cuda.jit(debug=True)(pyfunc)

    def test_from_dtype(self):
        rec = numpy_support.from_dtype(recordtype)
        self.assertEqual(rec.typeof('a'), types.float64)
        self.assertEqual(rec.typeof('b'), types.int32)
        self.assertEqual(rec.typeof('c'), types.complex64)
        if IS_PY3:
            self.assertEqual(rec.typeof('d'), types.UnicodeCharSeq(5))
        else:
            self.assertEqual(rec.typeof('d'), types.CharSeq(5))
        self.assertEqual(rec.offset('a'), recordtype.fields['a'][1])
        self.assertEqual(rec.offset('b'), recordtype.fields['b'][1])
        self.assertEqual(rec.offset('c'), recordtype.fields['c'][1])
        self.assertEqual(rec.offset('d'), recordtype.fields['d'][1])
        self.assertEqual(recordtype.itemsize, rec.size)

    def _test_set_equal(self, pyfunc, value, valuetype):
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp, valuetype))

        for i in range(self.refsample1d.size):
            expect = self.refsample1d.copy()
            pyfunc(expect, i, value)

            got = self.nbsample1d.copy()
            cfunc(got, i, value)

            # Match the entire array to ensure no memory corruption
            self.assertTrue(np.all(expect == got))

    def test_set_a(self):
        self._test_set_equal(set_a, 3.1415, types.float64)
        # Test again to check if coercion works
        self._test_set_equal(set_a, 3., types.float32)

    def test_set_b(self):
        self._test_set_equal(set_b, 123, types.int32)
        # Test again to check if coercion works
        self._test_set_equal(set_b, 123, types.float64)

    def test_set_c(self):
        self._test_set_equal(set_c, 43j, types.complex64)
        # Test again to check if coercion works
        self._test_set_equal(set_c, 43j, types.complex128)

    def test_set_record(self):
        pyfunc = set_record
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp, types.intp))

        test_indices = [(0, 1), (1, 2), (0, 2)]
        for i, j in test_indices:
            expect = self.refsample1d.copy()
            pyfunc(expect, i, j)

            got = self.nbsample1d.copy()
            cfunc(got, i, j)

            # Match the entire array to ensure no memory corruption
            self.assertEqual(expect[i], expect[j])
            self.assertEqual(got[i], got[j])
            self.assertTrue(np.all(expect == got))

    def test_record_with_array(self):
        '''
        Testing the use of a record containing an array
        '''
        nbval1 = self.nbsample1d4.copy()
        nbrecord1 = numpy_support.from_dtype(recordwitharray)
        cfunc = self.get_cfunc(record_write_array, (nbrecord1,))

        d_nbval1 = cuda.to_device(nbval1[0])
        # For copying the result back into, since we can't copy back into a
        # record type (it only provides a read-only buffer)
        result = np.zeros(1, dtype=nbval1.dtype)
        d_nbval1.copy_to_host(result) # Works OK

        # Call kernel
        cfunc(d_nbval1)

        d_nbval1.copy_to_host(result) # Launch Failed

        expected = self.nbsample1d4.copy()
        expected[0][0] = 2
        expected[0][1][0] = 3.0
        expected[0][1][1] = 4.0

        np.testing.assert_equal(expected[:1], result)


class TestRecordDtypeWithStructArrays(TestRecordDtype):
    '''
    Same as TestRecordDtype, but using structured arrays instead of recarrays.
    '''

    def _createSampleArrays(self):
        '''
        Two different versions of the data structures are required because Numba
        supports attribute access on structured arrays, whereas Numpy does not.

        However, the semantics of recarrays and structured arrays are equivalent
        for these tests so Numpy with recarrays can be used for comparison with
        Numba using structured arrays.
        '''

        self.refsample1d = np.recarray(3, dtype=recordtype)
        self.refsample1d2 = np.recarray(3, dtype=recordtype2)
        self.refsample1d3 = np.recarray(3, dtype=recordtype)
        self.refsample1d4 = np.recarray(3, dtype=recordwitharray)

        self.nbsample1d = np.zeros(3, dtype=recordtype)
        self.nbsample1d2 = np.zeros(3, dtype=recordtype2)
        self.nbsample1d3 = np.zeros(3, dtype=recordtype)
        self.nbsample1d4 = np.zeros(3, dtype=recordwitharray)

if __name__ == '__main__':
    unittest.main()
