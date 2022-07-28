import math
import numpy as np
from numba.cuda.mathimpl import get_unary_impl_for_fn_and_ty

# this is lazily initialized to avoid circular imports
_ufunc_db = None


def _lazy_init_db():
    global _ufunc_db

    if _ufunc_db is None:
        _ufunc_db = {}
        _fill_ufunc_db(_ufunc_db)


def get_ufunc_info(ufunc_key):
    _lazy_init_db()
    return _ufunc_db[ufunc_key]


def _fill_ufunc_db(ufunc_db):
    from numba.np.npyfuncs import _check_arity_and_homogeneity
    from numba.np.npyfuncs import np_complex_sin_impl

    def _np_real_sin_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        sin_impl = get_unary_impl_for_fn_and_ty(math.sin, sig.args[0])
        return sin_impl(context, builder, sig, args)

    _ufunc_db[np.sin] = {
        'f->f': _np_real_sin_impl,
        'd->d': _np_real_sin_impl,
        'F->F': np_complex_sin_impl,
        'D->D': np_complex_sin_impl,
    }
