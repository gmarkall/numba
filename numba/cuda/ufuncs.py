import math
import numpy as np
from numba.core import typing
from numba.cuda.mathimpl import (get_unary_impl_for_fn_and_ty,
                                 get_binary_impl_for_fn_and_ty)

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
    from numba.cpython import cmathimpl, mathimpl
    from numba.np.npyfuncs import _check_arity_and_homogeneity
    # FIXME: These could really have more efficient implementations for CUDA
    # that make use of libdevice
    from numba.np.npyfuncs import (np_complex_acosh_impl, np_complex_cos_impl,
                                   np_complex_sin_impl)

    def _np_real_sin_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        sin_impl = get_unary_impl_for_fn_and_ty(math.sin, sig.args[0])
        return sin_impl(context, builder, sig, args)

    def _np_real_cos_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        cos_impl = get_unary_impl_for_fn_and_ty(math.cos, sig.args[0])
        return cos_impl(context, builder, sig, args)

    def _np_real_tan_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        tan_impl = get_unary_impl_for_fn_and_ty(math.tan, sig.args[0])
        return tan_impl(context, builder, sig, args)

    def _np_real_asin_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        asin_impl = get_unary_impl_for_fn_and_ty(math.asin, sig.args[0])
        return asin_impl(context, builder, sig, args)

    def _np_real_acos_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        acos_impl = get_unary_impl_for_fn_and_ty(math.acos, sig.args[0])
        return acos_impl(context, builder, sig, args)

    def _np_real_atan_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        atan_impl = get_unary_impl_for_fn_and_ty(math.atan, sig.args[0])
        return atan_impl(context, builder, sig, args)

    def _np_real_atan2_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 2)
        atan2_impl = get_binary_impl_for_fn_and_ty(math.atan2, sig.args[0])
        return atan2_impl(context, builder, sig, args)

    def _np_real_hypot_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 2)
        atan2_impl = get_binary_impl_for_fn_and_ty(math.hypot, sig.args[0])
        return atan2_impl(context, builder, sig, args)

    def _np_real_sinh_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        sinh_impl = get_unary_impl_for_fn_and_ty(math.sinh, sig.args[0])
        return sinh_impl(context, builder, sig, args)

    def _np_complex_sinh_impl(context, builder, sig, args):
        # npymath does not provide a complex sinh. The code in funcs.inc.src
        # is translated here...
        _check_arity_and_homogeneity(sig, args, 1)

        ty = sig.args[0]
        fty = ty.underlying_float
        fsig1 = typing.signature(*[fty] * 2)
        x = context.make_complex(builder, ty, args[0])
        out = context.make_complex(builder, ty)
        xr = x.real
        xi = x.imag

        sxi = _np_real_sin_impl(context, builder, fsig1, [xi])
        shxr = _np_real_sinh_impl(context, builder, fsig1, [xr])
        cxi = _np_real_cos_impl(context, builder, fsig1, [xi])
        chxr = _np_real_cosh_impl(context, builder, fsig1, [xr])

        out.real = builder.fmul(cxi, shxr)
        out.imag = builder.fmul(sxi, chxr)

        return out._getvalue()

    def _np_real_cosh_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        cosh_impl = get_unary_impl_for_fn_and_ty(math.cosh, sig.args[0])
        return cosh_impl(context, builder, sig, args)

    def _np_complex_cosh_impl(context, builder, sig, args):
        # npymath does not provide a complex cosh. The code in funcs.inc.src
        # is translated here...
        _check_arity_and_homogeneity(sig, args, 1)

        ty = sig.args[0]
        fty = ty.underlying_float
        fsig1 = typing.signature(*[fty] * 2)
        x = context.make_complex(builder, ty, args[0])
        out = context.make_complex(builder, ty)
        xr = x.real
        xi = x.imag

        cxi = _np_real_cos_impl(context, builder, fsig1, [xi])
        chxr = _np_real_cosh_impl(context, builder, fsig1, [xr])
        sxi = _np_real_sin_impl(context, builder, fsig1, [xi])
        shxr = _np_real_sinh_impl(context, builder, fsig1, [xr])

        out.real = builder.fmul(cxi, chxr)
        out.imag = builder.fmul(sxi, shxr)

        return out._getvalue()

    def _np_real_tanh_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        tanh_impl = get_unary_impl_for_fn_and_ty(math.tanh, sig.args[0])
        return tanh_impl(context, builder, sig, args)

    def _np_complex_tanh_impl(context, builder, sig, args):
        # npymath does not provide complex tan functions. The code
        # in funcs.inc.src for tanh is translated here...
        _check_arity_and_homogeneity(sig, args, 1)

        ty = sig.args[0]
        fty = ty.underlying_float
        fsig1 = typing.signature(*[fty] * 2)
        ONE = context.get_constant(fty, 1.0)
        x = context.make_complex(builder, ty, args[0])
        out = context.make_complex(builder, ty)

        xr = x.real
        xi = x.imag
        si = _np_real_sin_impl(context, builder, fsig1, [xi])
        ci = _np_real_cos_impl(context, builder, fsig1, [xi])
        shr = _np_real_sinh_impl(context, builder, fsig1, [xr])
        chr_ = _np_real_cosh_impl(context, builder, fsig1, [xr])
        rs = builder.fmul(ci, shr)
        is_ = builder.fmul(si, chr_)
        rc = builder.fmul(ci, chr_)
        # Note: opposite sign for `ic` from code in funcs.inc.src
        ic = builder.fmul(si, shr)
        sqr_rc = builder.fmul(rc, rc)
        sqr_ic = builder.fmul(ic, ic)
        d = builder.fadd(sqr_rc, sqr_ic)
        inv_d = builder.fdiv(ONE, d)
        rs_rc = builder.fmul(rs, rc)
        is_ic = builder.fmul(is_, ic)
        is_rc = builder.fmul(is_, rc)
        rs_ic = builder.fmul(rs, ic)
        numr = builder.fadd(rs_rc, is_ic)
        numi = builder.fsub(is_rc, rs_ic)
        out.real = builder.fmul(numr, inv_d)
        out.imag = builder.fmul(numi, inv_d)

        return out._getvalue()

    def _np_real_asinh_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        asinh_impl = get_unary_impl_for_fn_and_ty(math.asinh, sig.args[0])
        return asinh_impl(context, builder, sig, args)

    def _np_real_acosh_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        acosh_impl = get_unary_impl_for_fn_and_ty(math.acosh, sig.args[0])
        return acosh_impl(context, builder, sig, args)

    def _np_real_atanh_impl(context, builder, sig, args):
        _check_arity_and_homogeneity(sig, args, 1)
        atanh_impl = get_unary_impl_for_fn_and_ty(math.atanh, sig.args[0])
        return atanh_impl(context, builder, sig, args)

    _ufunc_db[np.sin] = {
        'f->f': _np_real_sin_impl,
        'd->d': _np_real_sin_impl,
        'F->F': np_complex_sin_impl,
        'D->D': np_complex_sin_impl,
    }

    _ufunc_db[np.cos] = {
        'f->f': _np_real_cos_impl,
        'd->d': _np_real_cos_impl,
        'F->F': np_complex_cos_impl,
        'D->D': np_complex_cos_impl,
    }

    _ufunc_db[np.tan] = {
        'f->f': _np_real_tan_impl,
        'd->d': _np_real_tan_impl,
        'F->F': cmathimpl.tan_impl,
        'D->D': cmathimpl.tan_impl,
    }

    _ufunc_db[np.arcsin] = {
        'f->f': _np_real_asin_impl,
        'd->d': _np_real_asin_impl,
        'F->F': cmathimpl.asin_impl,
        'D->D': cmathimpl.asin_impl,
    }

    _ufunc_db[np.arccos] = {
        'f->f': _np_real_acos_impl,
        'd->d': _np_real_acos_impl,
        'F->F': cmathimpl.acos_impl,
        'D->D': cmathimpl.acos_impl,
    }

    _ufunc_db[np.arctan] = {
        'f->f': _np_real_atan_impl,
        'd->d': _np_real_atan_impl,
        'F->F': cmathimpl.atan_impl,
        'D->D': cmathimpl.atan_impl,
    }

    _ufunc_db[np.arctan2] = {
        'ff->f': _np_real_atan2_impl,
        'dd->d': _np_real_atan2_impl,
    }

    _ufunc_db[np.hypot] = {
        'ff->f': _np_real_hypot_impl,
        'dd->d': _np_real_hypot_impl,
    }

    _ufunc_db[np.sinh] = {
        'f->f': _np_real_sinh_impl,
        'd->d': _np_real_sinh_impl,
        'F->F': _np_complex_sinh_impl,
        'D->D': _np_complex_sinh_impl,
    }

    _ufunc_db[np.cosh] = {
        'f->f': _np_real_cosh_impl,
        'd->d': _np_real_cosh_impl,
        'F->F': _np_complex_cosh_impl,
        'D->D': _np_complex_cosh_impl,
    }

    _ufunc_db[np.tanh] = {
        'f->f': _np_real_tanh_impl,
        'd->d': _np_real_tanh_impl,
        'F->F': _np_complex_tanh_impl,
        'D->D': _np_complex_tanh_impl,
    }

    _ufunc_db[np.arcsinh] = {
        'f->f': _np_real_asinh_impl,
        'd->d': _np_real_asinh_impl,
        'F->F': cmathimpl.asinh_impl,
        'D->D': cmathimpl.asinh_impl,
    }

    _ufunc_db[np.arccosh] = {
        'f->f': _np_real_acosh_impl,
        'd->d': _np_real_acosh_impl,
        'F->F': np_complex_acosh_impl,
        'D->D': np_complex_acosh_impl,
    }

    _ufunc_db[np.arctanh] = {
        'f->f': _np_real_atanh_impl,
        'd->d': _np_real_atanh_impl,
        'F->F': cmathimpl.atanh_impl,
        'D->D': cmathimpl.atanh_impl,
    }

    ufunc_db[np.deg2rad] = {
        'f->f': mathimpl.radians_float_impl,
        'd->d': mathimpl.radians_float_impl,
    }

    ufunc_db[np.radians] = ufunc_db[np.deg2rad]

    ufunc_db[np.rad2deg] = {
        'f->f': mathimpl.degrees_float_impl,
        'd->d': mathimpl.degrees_float_impl,
    }

    ufunc_db[np.degrees] = ufunc_db[np.rad2deg]
