from numba import cffi_support, jit
import numpy as np
import cffi, ctypes

# Create the vectormaths module

defs = "void vsSin(int n, float* x, float* y);"
source = """\
void vsSin(int n, float* x, float* y) {
    int i;
    for (i=0; i<n; i++)
        y[i] = sin(x[i]);
}"""


ffi = cffi.FFI()
ffi.set_source('vectormaths', source)
ffi.cdef(defs, override=True)
ffi.compile()


# Import the compiled module

import vectormaths
cffi_support.register_module(vectormaths)
vsSin = vectormaths.lib.vsSin


# Method 1, works in Python, passes numpy arrays using casts.
# Pro: it actually works in Python
# Cons: it's a bit fiddly, may be difficult to support ffi.cast in Numba well

def python_version(x):
    y = np.empty_like(x)
    vsSin(len(x), ffi.cast('float *', x.ctypes.data),
        ffi.cast('float *', y.ctypes.data))
    return y

x = np.arange(10).astype(np.float32)
python_y = python_version(x)


# Method 2, works in Numba (depends on the arrays_to_ptrs branch):
# Pros: It can be made to work in Numba, code a bit cleaner
# Con: doesn't actually work without the @jit decorator

@jit(nopython=True)
def numba_version(x):
    y = np.empty_like(x)
    vsSin(len(x), ffi.from_buffer(x),
                  ffi.from_buffer(y))
    return y

x = np.arange(10).astype(np.float32)
numba_y = numba_version(x)


# These two versions do produce the same thing
assert np.all(python_y == numba_y)
