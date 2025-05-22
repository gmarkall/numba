"""
Test threadsafety for compiler.
These tests will cause segfault if fail.
"""
import threading
import random

import numpy as np

from numba import jit

from numba.tests.support import temp_directory, override_config
from numba.core import config
import unittest


def foo(n, v):
    return np.ones(n)


def ufunc_foo(a, b):
    return a + b


def gufunc_foo(a, b, out):
    out[0] = a + b



class TestThreadSafety(unittest.TestCase):

    def run_jit(self, **options):
        def runner():
            cfunc = jit(**options)(foo)

            return cfunc(4, 10)
        return runner

    def run_compile(self, fnlist):
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            def chooser():
                for _ in range(10):
                    fn = random.choice(fnlist)
                    fn()

            ths = [threading.Thread(target=chooser)
                   for i in range(4)]
            for th in ths:
                th.start()
            for th in ths:
                th.join()

    def test_concurrent_jit(self):
        self.run_compile([self.run_jit(nopython=True)])

    def test_concurrent_jit_cache(self):
        self.run_compile([self.run_jit(nopython=True, cache=True)])

    def test_concurrent_mix_use(self):
        self.run_compile([self.run_jit(nopython=True, cache=True),
                          self.run_jit(nopython=True)])


if __name__ == '__main__':
    unittest.main()
