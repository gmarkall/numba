import math
from numba.core import errors
from numba.core.imputils import Registry
from numba.types import float32, float64
from numba.cuda import libdevice
registry = Registry()
lower = registry.lower


#@lower(math.cos, types.f4)
#def lower_cos_float(context, builder, sig, args):
#    def cos(x):
#        return libdevice.cosf(x)

#    return context.compile_internal(builder, lambda x: libdevice.cosf(x), sig,
#    args)


#@lower(math.cos, types.f8)
#def lower_cos_double(context, builder, sig, args):
#    def cos(x):
#        return libdevice.cos(x)

#    return context.compile_internal(builder, lambda x: libdevice.cos(x), sig,
#    args)
#    return context.compile_internal(builder, cos, sig, args)


def impl_unary(key, ty, libfunc):
    def lower_unary_impl(context, builder, sig, args):
        cres = context.compile_subroutine(builder, lambda x: libfunc(x), sig)
        retty = sig.return_type
        got_retty = cres.signature.return_type
        if got_retty != retty:
            # This error indicates an error in *func* or the caller of this
            # method.
            raise errors.LoweringError(
                f'mismatching signature {got_retty} != {retty}.\n'
            )
        # Call into *func*
        status, res = context.call_internal_no_propagate(
            builder, cres.fndesc, sig, args,
        )

        return res

    lower(key, ty)(lower_unary_impl)


unarys = []
unarys += [('ceil', 'ceilf', math.ceil)]
unarys += [('floor', 'floorf', math.floor)]
unarys += [('fabs', 'fabsf', math.fabs)]
unarys += [('exp', 'expf', math.exp)]
unarys += [('expm1', 'expm1f', math.expm1)]
unarys += [('erf', 'erff', math.erf)]
unarys += [('erfc', 'erfcf', math.erfc)]
unarys += [('tgamma', 'tgammaf', math.gamma)]
unarys += [('lgamma', 'lgammaf', math.lgamma)]
unarys += [('sqrt', 'sqrtf', math.sqrt)]
unarys += [('log', 'logf', math.log)]
unarys += [('log10', 'log10f', math.log10)]
unarys += [('log1p', 'log1pf', math.log1p)]
unarys += [('acosh', 'acoshf', math.acosh)]
unarys += [('acos', 'acosf', math.acos)]
unarys += [('cos', 'cosf', math.cos)]
unarys += [('cosh', 'coshf', math.cosh)]
unarys += [('asinh', 'asinhf', math.asinh)]
unarys += [('asin', 'asinf', math.asin)]
unarys += [('sin', 'sinf', math.sin)]
unarys += [('sinh', 'sinhf', math.sinh)]
unarys += [('atan', 'atanf', math.atan)]
unarys += [('atanh', 'atanhf', math.atanh)]
unarys += [('tan', 'tanf', math.tan)]
unarys += [('tanh', 'tanhf', math.tanh)]

for fname64, fname32, key in unarys:
    impl32 = getattr(libdevice, fname32)
    impl64 = getattr(libdevice, fname64)
    impl_unary(key, float32, impl32)
    impl_unary(key, float64, impl64)
