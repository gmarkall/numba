from numba.core.descriptors import TargetDescriptor
from numba.core.options import DefaultOptionsMixin, TargetOptions
from .target import CUDATargetContext, CUDATypingContext


class CUDAOptionsMixin:
    opt = TargetOptions.Mapping('opt')
    lineinfo = TargetOptions.Mapping('lineinfo')


class CUDATargetOptions(DefaultOptionsMixin, CUDAOptionsMixin, TargetOptions):

    def finalize(self, flags, options):
        #breakpoint()
        flags.inherit_if_not_set('lineinfo')


class CUDATarget(TargetDescriptor):
    def __init__(self, name):
        self.options = CUDATargetOptions
        # The typing and target contexts are initialized only when needed -
        # this prevents an attempt to load CUDA libraries at import time on
        # systems that might not have them present.
        self._typingctx = None
        self._targetctx = None
        super().__init__(name)

    @property
    def typing_context(self):
        if self._typingctx is None:
            self._typingctx = CUDATypingContext()
        return self._typingctx

    @property
    def target_context(self):
        if self._targetctx is None:
            self._targetctx = CUDATargetContext(self._typingctx)
        return self._targetctx


cuda_target = CUDATarget('cuda')
