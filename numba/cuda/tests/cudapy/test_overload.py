from numba import cuda
from numba.core.extending import overload
from numba.cuda.testing import CUDATestCase, unittest


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


class TestOverload(CUDATestCase):
    def test_simple(self):
        @cuda.jit
        def cuda_foo():
            bar()
            baz()

        cuda_foo[1, 1]()
        cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
