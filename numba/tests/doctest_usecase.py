"""
Test that all docstrings are the same:

>>> len({f.__doc__ for f in (a, b, c, d)})
1
"""
from numba import njit


def a():
    """>>> x = 1"""
    return 1


@njit
def b():
    """>>> x = 1"""
    return 1
