"""
Implementation of some CFFI functions
"""

from __future__ import print_function, absolute_import, division

try:
    import cffi
except ImportError:
    pass

from numba.targets.imputils import implement, impl_attribute, Registry
from numba import types
from . import arrayobj

registry = Registry()

#@registry.register
#@impl_attribute(types.ffi, 'from_buffer')
#def from_buffer(context, builder, typ, value):
#    return value

@registry.register
@implement('from_buffer', types.ffi, types.Kind(types.Array))
def from_buffer(context, builder, sig, args):
    assert len(sig.args) == 1
    assert len(args) == 1
    [fromty] = sig.args
    [val] = args
    # Type inference should have prevented passing a buffer from an
    # array to a pointer of the wrong type
    assert fromty.dtype == sig.return_type.dtype
    ary = arrayobj.make_array(fromty)(context, builder, val)
    return ary.data
