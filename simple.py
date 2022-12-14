import os

os.environ['NUMBA_TRACE'] = '1'
os.environ['NUMBA_LLVM_REFPRUNE_PASS'] = '0'
os.environ['NUMBA_DUMP_LLVM'] = '1'
#os.environ['NUMBA_OPT'] = '0'

from numba import njit

@njit(_nrt=False)
def f(a, b):
    return a + b

print(f(1, 2))
