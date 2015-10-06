"""
Implementation of some CFFI functions
"""

from __future__ import print_function, absolute_import, division

from numba.targets.imputils import implement, Registry
from numba.typing.cffi_utils import ffi_module
from numba import types
from . import arrayobj

registry = Registry()

@registry.register
@implement('from_buffer', types.Kind(types.Array))
def from_buffer(context, builder, sig, args):
    # Type inference should have prevented passing a buffer from an
    # array to a pointer of the wrong type
    assert len(sig.args) == 1
    assert len(args) == 1
    [fromty] = sig.args
    [val] = args
    toty = sig.return_type
    assert fromty.dtype == toty.dtype
    ary = arrayobj.make_array(fromty)(context, builder, val)
    res = ary.data
    return res
    #Get form type
    # get to type
    #check theyre the same
    # make array struct from from
    # get the data pointer
    # return the data pointer

