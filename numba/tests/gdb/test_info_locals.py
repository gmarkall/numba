from numba import njit
from numba.tests.gdb_support import GdbMIDriver
from numba.tests.support import TestCase, needs_subprocess
import unittest


@njit(debug=True)
def foo(cond1, cond2):
    if cond1 and cond2:
        return 1
    else:
        return 2


@needs_subprocess
class Test(TestCase):

    def test(self):
        foo(120)
        driver = GdbMIDriver(__file__)
        driver.set_breakpoint(symbol="__main__::foo")
        driver.run() # will hit cpython symbol match
        driver.check_hit_breakpoint(number=1)
        driver.stack_list_variables(1)
        # We should not see variables for if conditions in the output - see
        # Numba PR #9888: https://github.com/numba/numba/pull/9888
        not_expected = "bool"
        driver.assert_output_regex(not_expected)
        driver.quit()


if __name__ == '__main__':
    unittest.main()
