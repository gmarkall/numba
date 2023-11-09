"""
Added for symmetry with the core API
"""

from numba.core.extending import intrinsic as _intrinsic
from functools import singledispatch

intrinsic = _intrinsic(target='cuda')


@singledispatch
def prepare_arg(val, ty):
    return val, ty
