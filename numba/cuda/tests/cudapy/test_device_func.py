import re
import types

import numpy as np

from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, int32
from numba.core.errors import TypingError


class TestDeviceFunc(CUDATestCase):

    def test_use_add2f(self):

        @cuda.jit(device=True)
        def add2f(a, b):
            return a + b

        def use_add2f(ary):
            i = cuda.grid(1)
            ary[i] = add2f(ary[i], ary[i])

        compiled = cuda.jit("void(float32[:])")(use_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def test_indirect_add2f(self):

        @cuda.jit(device=True)
        def add2f(a, b):
            return a + b

        @cuda.jit(device=True)
        def indirect(a, b):
            return add2f(a, b)

        def indirect_add2f(ary):
            i = cuda.grid(1)
            ary[i] = indirect(ary[i], ary[i])

        compiled = cuda.jit("void(float32[:])")(indirect_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def _check_cpu_dispatcher(self, add):
        @cuda.jit
        def add_kernel(ary):
            i = cuda.grid(1)
            ary[i] = add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    def test_cpu_dispatcher(self):
        # Test correct usage
        @jit
        def add(a, b):
            return a + b

        self._check_cpu_dispatcher(add)

    @skip_on_cudasim('not supported in cudasim')
    def test_cpu_dispatcher_invalid(self):
        # Test invalid usage
        # Explicit signature disables compilation, which also disable
        # compiling on CUDA.
        @jit('(i4, i4)')
        def add(a, b):
            return a + b

        # Check that the right error message is provided.
        with self.assertRaises(TypingError) as raises:
            self._check_cpu_dispatcher(add)
        msg = "Untyped global name 'add':.*using cpu function on device"
        expected = re.compile(msg)
        self.assertTrue(expected.search(str(raises.exception)) is not None)

    def test_cpu_dispatcher_other_module(self):
        @jit
        def add(a, b):
            return a + b

        mymod = types.ModuleType(name='mymod')
        mymod.add = add
        del add

        @cuda.jit
        def add_kernel(ary):
            i = cuda.grid(1)
            ary[i] = mymod.add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_ptx(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (int32, int32)
        cres = foo.compile_device(args)

        fname = cres.fndesc.mangled_name
        # Verify that the function name has "foo" in it as in the python name
        self.assertIn('foo', fname)

        ptx = foo.inspect_ptx(args)
        # Check that the compiled function name is in the PTX.
        self.assertIn(fname, ptx.decode('ascii'))

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_llvm(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (int32, int32)
        cres = foo.compile_device(args)

        fname = cres.fndesc.mangled_name
        # Verify that the function name has "foo" in it as in the python name
        self.assertIn('foo', fname)

        llvm = foo.inspect_llvm(args)
        # Check that the compiled function name is in the LLVM.
        self.assertIn(fname, llvm)

    @skip_on_cudasim('not supported in cudasim')
    def test_unsupported_eager_device(self):
        with self.assertRaises(NotImplementedError) as e:
            cuda.jit('int32(int32)', device=True)

        self.assertIn('Eager compilation of device functions is unsupported',
                      str(e.exception))

    @skip_on_cudasim('cudasim ignores casting by jit decorator signature')
    def test_device_casting(self):
        @cuda.jit('int32(int32, int32, int32, int32)')
        def rgba(r, g, b, a):
            return (((r & 0xFF) << 16) |
                    ((g & 0xFF) << 8) |
                    ((b & 0xFF) << 0) |
                    ((a & 0xFF) << 24))

        @cuda.jit
        def rgba_caller(x, channels):
            x[0] = rgba(channels[0], channels[1], channels[2], channels[3])

        x = cuda.device_array(1, dtype=np.int32)
        channels = cuda.to_device(np.asarray([1.0, 2.0, 3.0, 4.0],
                                             dtype=np.float32))

        rgba_caller[1, 1](x, channels)

        self.assertEqual(0x04010203, x[0])

    def test_kernel_then_device(self):
        # Presently fails because the overload of f that is a kernel gets used
        # as the overload for a device function when compiling f_caller
        @cuda.jit
        def f(x, y):
            x[0] = y[0]

        @cuda.jit
        def f_caller(x, y):
            f(x, y)

        one = np.ones(1)
        zero = np.zeros_like(one)
        y1 = cuda.to_device(one)
        y2 = cuda.to_device(one)
        x1 = cuda.to_device(zero)
        x2 = cuda.to_device(zero)

        f[1, 1](x1, y1)
        f_caller[1, 1](x2, y2)

        self.assertEqual(x1[0], y1[0])
        self.assertEqual(x2[0], y2[0])


if __name__ == '__main__':
    unittest.main()
