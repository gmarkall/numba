from __future__ import print_function, division, absolute_import

import sys

import numpy as np
from numba import jit, numpy_support, types
from numba import unittest_support as unittest
from numba.compiler import compile_isolated
from numba.utils import IS_PY3


def get_a(ary, i):
    return ary[i].a


def get_b(ary, i):
    return ary[i].b


def get_c(ary, i):
    return ary[i].c


def get_two_arrays_a(ary1, ary2, i):
    return ary1[i].a + ary2[i].a

def get_two_arrays_b(ary1, ary2, i):
    return ary1[i].b + ary2[i].b

def get_two_arrays_c(ary1, ary2, i):
    return ary1[i].c + ary2[i].c


def get_two_arrays_distinct(ary1, ary2, i):
    return ary1[i].a + ary2[i].f


def set_a(ary, i, v):
    ary[i].a = v


def set_b(ary, i, v):
    ary[i].b = v


def set_c(ary, i, v):
    ary[i].c = v


def set_record(ary, i, j):
    ary[i] = ary[j]


def get_record_a(rec, val):
    x = rec.a
    rec.a = val
    return x


def get_record_b(rec, val):
    x = rec.b
    rec.b = val
    return x


def get_record_c(rec, val):
    x = rec.c
    rec.c = val
    return x


def get_record_rev_a(val, rec):
    x = rec.a
    rec.a = val
    return x


def get_record_rev_b(val, rec):
    x = rec.b
    rec.b = val
    return x


def get_record_rev_c(val, rec):
    x = rec.c
    rec.c = val
    return x


def get_two_records_a(rec1, rec2):
    x = rec1.a + rec2.a
    return x


def get_two_records_b(rec1, rec2):
    x = rec1.b + rec2.b
    return x


def get_two_records_c(rec1, rec2):
    x = rec1.c + rec2.c
    return x


def get_two_records_distinct(rec1, rec2):
    x = rec1.a + rec2.f
    return x


def record_return(ary, i):
    return ary[i]


recordtype = np.dtype([('a', np.float64),
                       ('b', np.int32),
                       ('c', np.complex64),
                       ('d', (np.str, 5))])

recordtype2 = np.dtype([('e', np.int32),
                        ('f', np.float64)])


class TestRecordDtype(unittest.TestCase):

    def setUp(self):
        ary = np.recarray(3, dtype=recordtype)
        self.sample1d = ary

        for i in range(ary.size):
            x = i + 1
            ary[i].a = x / 2
            ary[i].b = x
            ary[i].c = x * 1j
            ary[i].d = "%d" % x

        ary2 = np.recarray(3, dtype=recordtype2)
        self.sample1d2 = ary2

        for i in range(ary2.size):
            x = i + 5
            ary2[i].e = x
            ary2[i].f = x / 2

        ary3 = np.recarray(3, dtype=recordtype)
        self.sample1d3 = ary3

        for i in range(ary3.size):
            x = i + 10
            ary3[i].a = x / 2
            ary3[i].b = x
            ary3[i].c = x * 1j
            ary3[i].d = "%d" % x

    def get_cfunc(self, pyfunc, argspec):
        # Need to keep a reference to the compile result for the
        # wrapper function object to remain valid (!)
        self.__cres = compile_isolated(pyfunc, argspec)
        return self.__cres.entry_point

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

    def _test_get_equal(self, pyfunc):
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp))
        for i in range(self.sample1d.size):
            self.assertEqual(pyfunc(self.sample1d, i), cfunc(self.sample1d, i))

    def test_get_a(self):
        self._test_get_equal(get_a)

    def test_get_b(self):
        self._test_get_equal(get_b)

    def test_get_c(self):
        self._test_get_equal(get_c)

    def _test_get_two_equal(self, pyfunc):
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], rec[:], types.intp))
        for i in range(self.sample1d.size):
            self.assertEqual(pyfunc(self.sample1d, self.sample1d3, i),
                              cfunc(self.sample1d, self.sample1d3, i))

    def test_get_two_a(self):
        self._test_get_two_equal(get_two_arrays_a)

    def test_get_two_b(self):
        self._test_get_two_equal(get_two_arrays_b)

    def test_get_two_c(self):
        self._test_get_two_equal(get_two_arrays_c)

    def _test_set_equal(self, pyfunc, value, valuetype):
        rec = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (rec[:], types.intp, valuetype))

        for i in range(self.sample1d.size):
            expect = self.sample1d.copy()
            pyfunc(expect, i, value)

            got = self.sample1d.copy()
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
            expect = self.sample1d.copy()
            pyfunc(expect, i, j)

            got = self.sample1d.copy()
            cfunc(got, i, j)

            # Match the entire array to ensure no memory corruption
            self.assertEqual(expect[i], expect[j])
            self.assertEqual(got[i], got[j])
            self.assertTrue(np.all(expect == got))


    def _test_record_args(self, revargs):
        """
        Testing scalar record value as argument
        """
        recval = self.sample1d.copy()[0]
        attrs = 'abc'
        valtypes = types.float64, types.int32, types.complex64
        values = 1.23, 123432, 132j
        old_refcnt = sys.getrefcount(recval)

        for attr, valtyp, val in zip(attrs, valtypes, values):
            expected = getattr(recval, attr)
            nbrecord = numpy_support.from_dtype(recordtype)

            # Test with a record as either the first argument or the second
            # argument (issue #870)
            if revargs:
                prefix = 'get_record_rev_'
                argtypes = (valtyp, nbrecord)
                args = (val, recval)
            else:
                prefix = 'get_record_'
                argtypes = (nbrecord, valtyp)
                args = (recval, val)

            pyfunc = globals()[prefix + attr]
            cfunc = self.get_cfunc(pyfunc, argtypes)

            got = cfunc(*args)
            self.assertEqual(expected, got)
            self.assertNotEqual(recval.a, got)
            del got, expected, args

        # Check for potential leaks (issue #441)
        self.assertEqual(sys.getrefcount(recval), old_refcnt)


    def test_record_args(self):
        self._test_record_args(False)


    def test_record_args_reverse(self):
        self._test_record_args(True)


    def test_two_records(self):
        '''
        Testing the use of two scalar records of the same type
        '''
        recval1 = self.sample1d.copy()[0]
        recval2 = self.sample1d.copy()[1]
        attrs = 'abc'
        valtypes = types.float64, types.int32, types.complex64

        for attr, valtyp in zip(attrs, valtypes):
            expected = getattr(recval1, attr) + getattr(recval2, attr)

            nbrecord = numpy_support.from_dtype(recordtype)
            pyfunc = globals()['get_two_records_' + attr]
            cfunc = self.get_cfunc(pyfunc, (nbrecord, nbrecord))

            got = cfunc(recval1, recval2)
            self.assertEqual(expected, got)

    def test_two_distinct_records(self):
        '''
        Testing the use of two scalar records of differing type
        '''
        recval1 = self.sample1d.copy()[0]
        recval2 = self.sample1d2.copy()[0]
        expected = recval1.a + recval2.f

        nbrecord1 = numpy_support.from_dtype(recordtype)
        nbrecord2 = numpy_support.from_dtype(recordtype2)
        cfunc = self.get_cfunc(get_two_records_distinct, (nbrecord1, nbrecord2))

        got = cfunc(recval1, recval2)
        self.assertEqual(expected, got)

    def test_record_return(self):
        """
        Testing scalar record value as return value.
        We can only return a copy of the record.
        """
        pyfunc = record_return
        recty = numpy_support.from_dtype(recordtype)
        cfunc = self.get_cfunc(pyfunc, (recty[:], types.intp))

        attrs = 'abc'
        indices = [0, 1, 2]
        for index, attr in zip(indices, attrs):
            ary = self.sample1d.copy()
            old_refcnt = sys.getrefcount(ary)
            res = cfunc(ary, index)
            self.assertEqual(ary[index], res)
            # Prove that this is a by-value copy
            setattr(res, attr, 0)
            self.assertNotEqual(ary[index], res)
            del res
            # Check for potential leaks
            self.assertEqual(sys.getrefcount(ary), old_refcnt)


class TestRecordDtypeWithDispatcher(TestRecordDtype):
    """
    Same as TestRecordDtype, but stressing the Dispatcher's type dispatch
    mechanism (issue #384).
    """

    def get_cfunc(self, pyfunc, argspec):
        return jit(argspec, nopython=True)(pyfunc)


if __name__ == '__main__':
    unittest.main()
