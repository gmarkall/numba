from numba.core import utils
from numba.core.descriptors import NestableTargetDescriptor, NestedContext
from numba.core.options import TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CUDATargetOptions(TargetOptions):
    pass


class CUDATarget(NestableTargetDescriptor):
    options = CUDATargetOptions
    _nested = NestedContext()

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return CUDATypingContext()

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return CUDATargetContext(self.typing_context)


cuda_target = CUDATarget('cuda')
