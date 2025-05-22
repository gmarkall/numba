import warnings
import unittest
from contextlib import contextmanager

from numba import jit
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning, NumbaWarning)
from numba.tests.support import TestCase, needs_setuptools


@contextmanager
def _catch_numba_deprecation_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore", category=NumbaWarning)
        warnings.simplefilter("always", category=NumbaDeprecationWarning)
        yield w


class TestDeprecation(TestCase):

    def check_warning(self, warnings, expected_str, category, check_rtd=True):
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, category)
        self.assertIn(expected_str, str(warnings[0].message))
        if check_rtd:
            self.assertIn("https://numba.readthedocs.io",
                          str(warnings[0].message))

    @TestCase.run_test_in_subprocess
    def test_explicit_false_nopython_kwarg(self):
        # tests that explicitly setting `nopython=False` in @jit raises a
        # warning about it doing nothing.
        with _catch_numba_deprecation_warnings() as w:

            @jit(nopython=False)
            def foo():
                pass

            foo()

            msg = "The keyword argument 'nopython=False' was supplied"
            self.check_warning(w, msg, NumbaDeprecationWarning, check_rtd=False)

    @TestCase.run_test_in_subprocess
    def test_reflection_of_mutable_container(self):
        # tests that reflection in list/set warns
        def foo_list(a):
            return a.append(1)

        def foo_set(a):
            return a.add(1)

        for f in [foo_list, foo_set]:
            container = f.__name__.strip('foo_')
            inp = eval(container)([10, ])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore", category=NumbaWarning)
                warnings.simplefilter("always",
                                      category=NumbaPendingDeprecationWarning)
                jit(nopython=True)(f)(inp)
                self.assertEqual(len(w), 1)
                self.assertEqual(w[0].category, NumbaPendingDeprecationWarning)
                warn_msg = str(w[0].message)
                msg = ("Encountered the use of a type that is scheduled for "
                       "deprecation")
                self.assertIn(msg, warn_msg)
                msg = ("\'reflected %s\' found for argument" % container)
                self.assertIn(msg, warn_msg)
                self.assertIn("https://numba.readthedocs.io", warn_msg)

    @needs_setuptools
    @TestCase.run_test_in_subprocess
    def test_pycc_module(self):
        # checks import of module warns

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always",
                                  category=NumbaPendingDeprecationWarning)
            import numba.pycc # noqa: F401

            expected_str = ("The 'pycc' module is pending deprecation.")
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)

    @needs_setuptools
    @TestCase.run_test_in_subprocess
    def test_pycc_CC(self):
        # check the most commonly used functionality (CC) warns

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always",
                                  category=NumbaPendingDeprecationWarning)
            from numba.pycc import CC # noqa: F401

            expected_str = ("The 'pycc' module is pending deprecation.")
            self.check_warning(w, expected_str, NumbaPendingDeprecationWarning)


if __name__ == '__main__':
    unittest.main()
