from numba import cgutils, types
from numba.targets.imputils import builtin_attr, impl_attribute

def make_interval():
    """
    Return the Structure representation of an interval
    """

    # This structure should be kept in sync with Numba_adapt_interval()
    # in _helperlib.c.
    class IntervalTemplate(cgutils.Structure):
        _fields = [('lo', types.float64),
                   ('hi', types.float64),
                  ]

    return IntervalTemplate

@builtin_attr
@impl_attribute(types.Kind(types.IntervalType), 'lo', types.float64)
def interval_lo(context, builder, typ, value):
    ivty = make_interval()
    iv = ivty(context, builder, value)
    return iv.lo

@builtin_attr
@impl_attribute(types.Kind(types.IntervalType), 'hi', types.float64)
def interval_hi(context, builder, typ, value):
    ivty = make_interval()
    iv = ivty(context, builder, value)
    return iv.hi
