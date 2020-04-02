from enum import Enum
from numba.core.extending import register_model, models
from numba.core import types
from numba.cuda.types import Dim3, ThreadGroup, ThreadBlock, uint24


@register_model(Dim3)
class Dim3Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.int32),
            ('y', types.int32),
            ('z', types.int32)
        ]
        super().__init__(dmm, fe_type, members)


@register_model(ThreadGroup)
@register_model(ThreadBlock)
class ThreadGroupModel(models.StructModel):
    # A simplification of the type in CUDA C/C++. We don't use a union, simply
    # a representation the same as thread_group.coalesced. (need some more
    # notes on why this is enough).
    def __init__(self, dmm, fe_type):
        members = [
            ('type', types.uint8),
            ('size', uint24),
            ('mask', types.uint32)
        ]
        super().__init__(dmm, fe_type, members)


class GroupType(Enum):
    CoalescedTile = 0
    Coalesced = 1
    ThreadBlock = 2
    Grid = 3
    MultiGrid = 4
