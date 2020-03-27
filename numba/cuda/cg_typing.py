import collections

from enum import Enum
from numba.core import cgutils, types
from numba.core.extending import (type_callable, models, register_model,
                                  lower_builtin)


class ThreadGroupType(types.Type):
    def __init__(self):
        super().__init__(name='thread_group')


thread_group_type = ThreadGroupType()


def this_thread_block():
    pass


@type_callable(this_thread_block)
def type_this_thread_block(context):
    def typer():
        return thread_group_type
    return typer


tg_coalesced_py = collections.namedtuple('coalesced', ('size',
                                                       'mask'))
tg_coalesced = types.NamedTuple((types.uint32, types.uint32), tg_coalesced_py)


@register_model(ThreadGroupType)
class ThreadGroupModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('type', types.uint8),
            ('coalesced', tg_coalesced),
            ('buffer', types.UniTuple(types.uintp, 2))
        ]
        super().__init__(dmm, fe_type, members)


@lower_builtin(this_thread_block)
def impl_this_thread_group(context, builder, sig, args):
    typ = sig.return_type
    tg = cgutils.create_struct_proxy(typ)(context, builder)
    return tg._getvalue()


class GroupType(Enum):
    CoalescedTile = 0
    Coalesced = 1
    ThreadBlock = 2
    Grid = 3
    MultiGrid = 4
