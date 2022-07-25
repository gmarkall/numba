import sys

from numba import cuda, njit
from numba.cuda.testing import CUDATestCase
from numba.cuda.tests.cudapy.usecases import CPUUseCase, CUDAUseCase


# Using the same function as a cached CPU and CUDA-jitted function

def target_shared_assign(r, x):
    r[()] = x[()]


assign_cuda_kernel = cuda.jit(cache=True)(target_shared_assign)
assign_cuda = CUDAUseCase(assign_cuda_kernel)
assign_cpu_jitted = njit(cache=True)(target_shared_assign)
assign_cpu = CPUUseCase(assign_cpu_jitted)


class _TestModule(CUDATestCase):
    """
    Tests for functionality of this module's functions.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    """

    def check_module(self, mod):
        self.assertPreciseEqual(mod.assign_cpu(5), 5)
        self.assertPreciseEqual(mod.assign_cpu(5.5), 5.5)
        self.assertPreciseEqual(mod.assign_cuda(5), 5)
        self.assertPreciseEqual(mod.assign_cuda(5.5), 5.5)


def self_test():
    mod = sys.modules[__name__]
    _TestModule().check_module(mod)
