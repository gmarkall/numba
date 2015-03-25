from __future__ import print_function

from numba import jit
from numba.types import FixedLenCharSeq, boolean
from numba.tests.support import unittest


charseq = FixedLenCharSeq(3)

@jit(boolean(charseq), nopython=True)
def f(chars):
    return chars == 'ABC'


class TestCharSeq(unittest.TestCase):
    def test_eq(self):
        self.assertTrue(f('ABC'))
        self.assertFalse(f('DEF'))


if __name__ == "__main__":
    unittest.main()
