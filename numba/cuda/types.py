from numba.core import types


class Dim3(types.Type):
    def __init__(self):
        super().__init__(name='Dim3')


dim3 = Dim3()


class ThreadGroup(types.Type):
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', 'ThreadGroup')
        super().__init__(name=name, *args, **kwargs)


class ThreadBlock(ThreadGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(name='ThreadBlock', *args, **kwargs)


class CoalescedGroup(ThreadGroup):
    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', 'CoalescedGroup')
        super().__init__(name=name, *args, **kwargs)


class CoalescedTile(CoalescedGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(name='CoalescedTile', *args, **kwargs)


thread_group = ThreadGroup()
thread_block = ThreadBlock()
coalesced_group = CoalescedGroup()
coalesced_tile = CoalescedTile()


# Used in the data model for thread groups
uint24 = types.Integer('uint24')
