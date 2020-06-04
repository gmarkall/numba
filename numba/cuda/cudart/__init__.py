"""CUDA Runtime

- Runtime API binding

"""
from numba.core import config
assert not config.ENABLE_CUDASIM, 'Cannot use real runtime API with simulator'
