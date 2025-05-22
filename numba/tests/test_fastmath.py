import math
import numpy as np

from numba import njit
import unittest


class TestFastMath(unittest.TestCase):
    def test_jit(self):
        def foo(x):
            return x + math.sin(x)
        fastfoo = njit(fastmath=True)(foo)
        slowfoo = njit(foo)
        self.assertEqual(fastfoo(0.5), slowfoo(0.5))
        fastllvm = fastfoo.inspect_llvm(fastfoo.signatures[0])
        slowllvm = slowfoo.inspect_llvm(slowfoo.signatures[0])
        # Ensure fast attribute in fast version only
        self.assertIn('fadd fast', fastllvm)
        self.assertIn('call fast', fastllvm)
        self.assertNotIn('fadd fast', slowllvm)
        self.assertNotIn('call fast', slowllvm)

    def test_jit_subset_behaviour(self):
        def foo(x, y):
            return (x - y) + y
        fastfoo = njit(fastmath={'reassoc', 'nsz'})(foo)
        slowfoo = njit(fastmath={'reassoc'})(foo)
        self.assertEqual(fastfoo(0.5, np.inf), 0.5)
        self.assertTrue(np.isnan(slowfoo(0.5, np.inf)))

    def test_jit_subset_code(self):
        def foo(x):
            return x + math.sin(x)
        fastfoo = njit(fastmath={'reassoc', 'nsz'})(foo)
        slowfoo = njit()(foo)
        self.assertEqual(fastfoo(0.5), slowfoo(0.5))
        fastllvm = fastfoo.inspect_llvm(fastfoo.signatures[0])
        slowllvm = slowfoo.inspect_llvm(slowfoo.signatures[0])
        # Ensure fast attributes in fast version only
        self.assertNotIn('fadd fast', slowllvm)
        self.assertNotIn('call fast', slowllvm)
        self.assertNotIn('fadd reassoc nsz', slowllvm)
        self.assertNotIn('call reassoc nsz', slowllvm)
        self.assertNotIn('fadd nsz reassoc', slowllvm)
        self.assertNotIn('call nsz reassoc', slowllvm)
        self.assertTrue(
            ('fadd nsz reassoc' in fastllvm) or
            ('fadd reassoc nsz' in fastllvm),
            fastllvm
        )
        self.assertTrue(
            ('call nsz reassoc' in fastllvm) or
            ('call reassoc nsz' in fastllvm),
            fastllvm
        )

    def test_jit_subset_errors(self):
        with self.assertRaises(ValueError) as raises:
            njit(fastmath={'spqr'})(lambda x: x + 1)(1)
        self.assertIn(
            "Unrecognized fastmath flags:",
            str(raises.exception),
        )

        with self.assertRaises(ValueError) as raises:
            njit(fastmath={'spqr': False})(lambda x: x + 1)(1)
        self.assertIn(
            'Unrecognized fastmath flags:',
            str(raises.exception),
        )

        with self.assertRaises(ValueError) as raises:
            njit(fastmath=1337)(lambda x: x + 1)(1)
        self.assertIn(
            'Expected fastmath option(s) to be',
            str(raises.exception),
        )


if __name__ == '__main__':
    unittest.main()
