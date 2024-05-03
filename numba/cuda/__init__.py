# Here we don't use moved_cuda_module() because we want to keep
# numba.cuda.tests from the main Numba repo, so that we can always run the
# tests from within Numba using the numba.cuda.tests module, rather than
# numba_cuda.tests - using moved_cuda_module() would shadow numba.cuda.tests
# with numba_cuda.tests.

from numba import runtests
from numba.core import config

if config.ENABLE_CUDASIM:
    from numba_cuda.simulator_init import *
else:
    from numba_cuda.device_init import *
    from numba_cuda.device_init import _auto_device

from numba_cuda.compiler import (compile, compile_for_current_device,
                                 compile_ptx, compile_ptx_for_current_device)

def test(*args, **kwargs):
    if not is_available():
        raise RuntimeError("CUDA is not available")

    main_args = ["<main>", "numba_cuda.tests", '-v', '-m']
    return runtests._main(main_args, *args, **kwargs)
