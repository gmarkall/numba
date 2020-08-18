import numpy as np
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase
from numba import cuda
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

def pyfunc(%(pyargs)s):
    ret = libdevice.%(func)s(%(funcargs)s)
    %(retvars)s = ret
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

        funcargsstr = ", ".join(['a%d' % i for i, arg in enumerate(args) if not
                                 arg.is_ptr])

        if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
            pyargsstr = ", ".join(['r%d' % i for i in
                                   range(len(sig.return_type))])
            pyargsstr += ", " + funcargsstr
            retvarsstr = ", ".join(['r%d[0]' % i for i in
                                    range(len(sig.return_type))])
        else:
            pyargsstr = "r0, " + funcargsstr
            retvarsstr = "r0[0]"
        d = { 'func': apiname, 'pyargs': pyargsstr, 'funcargs': funcargsstr,
              'retvars': retvarsstr }
        code = function_template % d
        print(code)
        locals = {}
        exec(code, globals(), locals)
        pyfunc = locals['pyfunc']

        pyargs = [ arg.ty for arg in args if not arg.is_ptr ]
        if isinstance(sig.return_type, (types.Tuple, types.UniTuple)):
            pyreturns = [ret[::1] for ret in sig.return_type]
            pyargs = pyreturns + pyargs
        else:
            pyargs.insert(0, retty[::1])
        print(pyargs)

        ptx, resty = compile_ptx(pyfunc, pyargs)

        print(ptx)

    return _test_call_functions


for libname in functions:
    setattr(TestLibdeviceCompilation, 'test_%s' % libname,
            make_test_call(libname))


def test_made(self):
    print("yep!")


setattr(TestLibdevice, 'test_made', test_made)


if __name__ == '__main__':
    unittest.main()
