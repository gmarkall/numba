# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import sys
import os
import re
import unittest

import numpy as np

from numba import set_num_threads, config, threading_layer
from numba.tests.support import TestCase, tag
from numba.tests.test_parallel_backend import TestInSubprocess


class TestNumThreads(TestCase):
    _numba_parallel_test_ = False

    def setUp(self):
        # Make sure the num_threads is set to the max. This also makes sure
        # the threads are launched.
        set_num_threads(config.NUMBA_NUM_THREADS)

    def check_mask(self, expected, result):
        # There's no guarantee that TBB will use a full mask worth of
        # threads if it deems it inefficient to do so
        if threading_layer() == 'tbb':
            self.assertTrue(np.all(result <= expected))
        elif threading_layer() in ('omp', 'workqueue'):
            np.testing.assert_equal(expected, result)
        else:
            assert 0, 'unreachable'


class TestNumThreadsBackends(TestInSubprocess, TestCase):
    _class = TestNumThreads
    _DEBUG = False

    # 1 is mainly here to ensure tests skip correctly
    num_threads = [i for i in [1, 2, 4, 8, 16] if i <= config.NUMBA_NUM_THREADS]

    def run_test_in_separate_process(self, test, threading_layer, num_threads):
        env_copy = os.environ.copy()
        env_copy['NUMBA_THREADING_LAYER'] = str(threading_layer)
        env_copy['NUMBA_NUM_THREADS'] = str(num_threads)
        cmdline = [sys.executable, "-m", "numba.runtests", "-v", test]
        return self.run_cmd(cmdline, env_copy)

    @classmethod
    def _inject(cls, name, backend, backend_guard, num_threads):
        themod = cls.__module__
        thecls = cls._class.__name__
        injected_method = '%s.%s.%s' % (themod, thecls, name)

        def test_template(self):
            o, e = self.run_test_in_separate_process(injected_method, backend,
                                                     num_threads)
            if self._DEBUG:
                print('stdout:\n "%s"\n stderr:\n "%s"' % (o, e))
            # If the test was skipped in the subprocess, then mark this as a
            # skipped test.
            m = re.search(r"\.\.\. skipped '(.*?)'", e)
            if m is not None:
                self.skipTest(m.group(1))
            self.assertIn('OK', e)
            self.assertTrue('FAIL' not in e)
            self.assertTrue('ERROR' not in e)

        injected_test = "%s_%s_%s_threads" % (name[1:], backend, num_threads)
        setattr(cls, injected_test,
                tag('long_running')(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        for name in cls._class.__dict__.copy():
            for backend, backend_guard in cls.backends.items():
                for num_threads in cls.num_threads:
                    if not name.startswith('_test_'):
                        continue
                    cls._inject(name, backend, backend_guard, num_threads)


TestNumThreadsBackends.generate()

if __name__ == '__main__':
    unittest.main()
