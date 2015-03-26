from __future__ import print_function

from numba import jit
from numba.tests.support import unittest


@jit(nopython=True)
def f(chars):
    return chars == 'ABC'

@jit(nopython=True)
def f1():
    return 'ABC' == 'ABC'

@jit(nopython=True)
def f2():
    return 'ABC' == 'AB'

@jit(nopython=True)
def f3():
    return 'ABC' == 'ABCD'

@jit(nopython=True)
def f4(chars1, chars2):
    return chars1 == chars2

class TestCharSeq(unittest.TestCase):
    def test_eq(self):
        self.assertTrue(f('ABC'))
        self.assertFalse(f('DEF'))
        self.assertFalse(f('AB'))
        self.assertFalse(f('ABCD'))

        self.assertTrue(f1())
        self.assertFalse(f2())
        self.assertFalse(f3())

        self.assertTrue(f4('ABC', 'ABC'))
        self.assertFalse(f4('ABC', 'ABD'))
        self.assertFalse(f4('ABC', 'AB'))
        self.assertFalse(f4('ABC', 'ABCD'))

if __name__ == "__main__":
    unittest.main()
