from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba.errors import TypingError
from numba import jit, types


a0 = np.array(42)

s1 = np.int32(64)

a1 = np.arange(12)
a2 = a1[::2]
a3 = a1.reshape((3, 4)).T

dt = np.dtype([('x', np.int8), ('y', 'S3')])

a4 = np.arange(32, dtype=np.int8).view(dt)
a5 = a4[::-2]

# A recognizable data string
a6 = np.frombuffer(b"XXXX_array_contents_XXXX", dtype=np.float32)


def getitem0(i):
    return a0[()]


def getitem1(i):
    return a1[i]


def getitem2(i):
    return a2[i]


def getitem3(i):
    return a3[i]


def getitem4(i):
    return a4[i]


def getitem5(i):
    return a5[i]


def getitem6(i):
    return a6[i]


def use_arrayscalar_const():
    return s1


def write_to_global_array():
    myarray[0] = 1


class TestConstantArray(unittest.TestCase):
    """
    Test array constants.
    """

    def check_array_const(self, pyfunc):
        cres = compile_isolated(pyfunc, (types.int32,))
        cfunc = cres.entry_point
        for i in [0, 1, 2]:
            np.testing.assert_array_equal(pyfunc(i), cfunc(i))

    def test_array_const_0d(self):
        self.check_array_const(getitem0)

    def test_array_const_1d_contig(self):
        self.check_array_const(getitem1)

    def test_array_const_1d_noncontig(self):
        self.check_array_const(getitem2)

    @unittest.skip('PYPY FIXME - mysterious death')
    def test_array_const_2d(self):
        self.check_array_const(getitem3)

    def test_record_array_const_contig(self):
        self.check_array_const(getitem4)

    def test_record_array_const_noncontig(self):
        self.check_array_const(getitem5)

    def test_array_const_alignment(self):
        """
        Issue #1933: the array declaration in the LLVM IR must have
        the right alignment specified.
        """
        sig = (types.intp,)
        cfunc = jit(sig, nopython=True)(getitem6)
        ir = cfunc.inspect_llvm(sig)
        for line in ir.splitlines():
            if 'XXXX_array_contents_XXXX' in line:
                self.assertIn("constant [24 x i8]", line)  # sanity check
                # Should be the ABI-required alignment for float32
                # on most platforms...
                self.assertIn(", align 4", line)
                break
        else:
            self.fail("could not find array declaration in LLVM IR")

    def test_arrayscalar_const(self):
        pyfunc = use_arrayscalar_const
        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point

        self.assertEqual(pyfunc(), cfunc())

    def test_write_to_global_array(self):
        pyfunc = write_to_global_array
        with self.assertRaises(TypingError):
            compile_isolated(pyfunc, ())

    def test_issue_1850(self):
        """
        This issue is caused by an unresolved bug in numpy since version 1.6.
        See numpy GH issue #3147.
        """
        constarr = np.array([86])

        def pyfunc():
            return constarr[0]

        cres = compile_isolated(pyfunc, ())
        out = cres.entry_point()
        self.assertEqual(out, 86)


if __name__ == '__main__':
    unittest.main()
