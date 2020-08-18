import numpy as np
from numba.core.typing import signature
from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda, void
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature


def use_sincos(s, c, x):
    i = cuda.grid(1)

    if i < len(x):
        sr, cr = libdevice.sincos(x[i])
        s[i] = sr
        c[i] = cr


function_template = """\
from numba.cuda import libdevice

def pyfunc(%(args)s):
    ret = libdevice.%(func)s(%(args)s)
"""


class TestLibdevice(CUDATestCase):

    def test_sincos(self):
        arr = np.arange(100, dtype=np.float64)
        sres = np.zeros_like(arr)
        cres = np.zeros_like(arr)

        cufunc = cuda.jit(use_sincos)
        cufunc[4, 32](sres, cres, arr)

        np.testing.assert_allclose(np.cos(arr), cres)
        np.testing.assert_allclose(np.sin(arr), sres)


class TestLibdeviceCompilation(unittest.TestCase):
    pass


def make_test_call(libname):
    def _test_call_functions(self):
        # Strip off '__nv_' from libdevice name to get Python name
        apiname = libname[5:]
        apifunc = getattr(libdevice, libname[5:])
        retty, args = functions[libname]
        sig = create_signature(retty, args)
        print(apifunc, sig)

        argsstr = ", ".join(['a%s' % i for i, arg in enumerate(args) if not
                             arg.is_ptr])
        d = { 'func': apiname, 'args': argsstr }
        print(d)
        code = function_template % d
        print(code)
        locals = {}
        exec(code, globals(), locals)
        pyfunc = locals['pyfunc']
        print(pyfunc)

        pyargs = [ arg.ty for arg in args if not arg.is_ptr ]
        pysig = signature(void, *pyargs)
        print(pysig)

        #cufunc = cuda.jit(pysig)(pyfunc)
        #print(cufunc)
        ptx, resty = compile_ptx(pyfunc, pyargs)

        print(ptx)

    return _test_call_functions


for libname in functions:
    print("Doing", libname)
    setattr(TestLibdeviceCompilation, 'test_%s' % libname,
            make_test_call(libname))


def test_made(self):
    print("yep!")


setattr(TestLibdevice, 'test_made', test_made)


if __name__ == '__main__':
    unittest.main()
