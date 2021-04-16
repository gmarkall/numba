from numba import njit, types
from numba.extending import overload, intrinsic
from numba.core.extending_hardware import dispatcher_registry, hardware_registry
from numba.core import decorators

from numba import cuda
from numba.cuda.compiler import Dispatcher as CUDADispatcher

dispatcher_registry[hardware_registry["cuda"]] = CUDADispatcher

def cuda_jit_device(*args, **kwargs):
    kwargs['device'] = True
    return cuda.jit(*args, **kwargs)

decorators.jit_registry[hardware_registry["cuda"]] = cuda_jit_device

# ------------------- GET THIS TO WORK

def bar():
    pass

def baz():
    pass

@overload(bar, hardware='generic')
def ol_bar():
    def impl():
        print("Generic bar")
    return impl

@overload(baz, hardware='cuda')
def ol_baz_cuda():
    def impl():
        print("CUDA baz")
    return impl

@overload(baz, hardware='cpu')
def ol_baz_cpu():
    def impl():
        print("CPU baz")
    return impl


@njit
def cpu_foo():
    bar()
    baz()

@cuda.jit
def cuda_foo():
    bar()
    baz()

print("CPU FOO")
cpu_foo()

print("CUDA FOO")
cuda_foo[1, 1]()

cuda.synchronize()
