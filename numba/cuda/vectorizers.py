from numba import cuda
from numba.core import sigutils
from numba.cuda import dispatcher
from numba.cuda.compiler import compile_device
from numba.np.ufunc import deviceufunc

vectorizer_stager_source = '''
def __vectorized_{name}({args}, __out__):
    __tid__ = __cuda__.grid(1)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
'''


class CUDAVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        argtypes, return_type = sigutils.normalize_signature(sig)
        cudevfn = compile_device(self.pyfunc, return_type, argtypes)
        return cudevfn, cudevfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({'__cuda__': cuda,
                     '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return cuda.jit(fnobj)

    def build_ufunc(self):
        return dispatcher.CUDAUFuncDispatcher(self.kernelmap)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source


# ------------------------------------------------------------------------------
# Generalized CUDA ufuncs

_gufunc_stager_source = '''
def __gufunc_{name}({args}):
    __tid__ = __cuda__.grid(1)
    if __tid__ < {checkedarg}:
        __core__({argitems})
'''


class CUDAGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):
    def build_ufunc(self):
        engine = deviceufunc.GUFuncEngine(self.inputsig, self.outputsig)
        return dispatcher.CUDAGenerializedUFunc(kernelmap=self.kernelmap,
                                                engine=engine)

    def _compile_kernel(self, fnobj, sig):
        return cuda.jit(sig)(fnobj)

    @property
    def _kernel_template(self):
        return _gufunc_stager_source

    def _get_globals(self, sig):
        argtypes, return_type = sigutils.normalize_signature(sig)
        corefn = compile_device(self.pyfunc, return_type, argtypes)
        glbls = self.py_func.__globals__.copy()
        glbls.update({'__cuda__': cuda,
                      '__core__': corefn})
        return glbls
