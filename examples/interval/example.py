from interval import Interval
from numba import njit
from numba.types import bool_, interval_type, float64

@njit(bool_(interval_type, float64))
def inside(interval, x):
    '''
    Check if a value is within an interval
    '''
    return interval.lo <= x < interval.hi

i = Interval(0, 2)
v = 1.0
r = inside(i, v)
print("%s is inside %s: %s" % (v, i, r))
