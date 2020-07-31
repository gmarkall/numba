import math
from numba.core import typing
from numba.core.imputils import Registry
from numba.types import float32, float64
from numba.cuda import libdevice

registry = Registry()
lower = registry.lower


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

binarys = []
binarys += [('copysign', 'copysignf', math.copysign)]
binarys += [('atan2', 'atan2f', math.atan2)]
binarys += [('pow', 'powf', math.pow)]
binarys += [('fmod', 'fmodf', math.fmod)]
binarys += [('hypot', 'hypotf', math.hypot)]


def impl_unary(key, ty, libfunc):
    def lower_unary_impl(context, builder, sig, args):
        libfunc_impl = context.get_function(libfunc, typing.signature(ty, ty))
        return libfunc_impl(builder, args)

    lower(key, ty)(lower_unary_impl)


def impl_binary(key, ty, libfunc):
    def lower_binary_impl(context, builder, sig, args):
        libfunc_impl = context.get_function(libfunc,
                                            typing.signature(ty, ty, ty))
        return libfunc_impl(builder, args)

    lower(key, ty, ty)(lower_binary_impl)


for fname64, fname32, key in unarys:
    impl32 = getattr(libdevice, fname32)
    impl64 = getattr(libdevice, fname64)
    impl_unary(key, float32, impl32)
    impl_unary(key, float64, impl64)


for fname64, fname32, key in binarys:
    impl32 = getattr(libdevice, fname32)
    impl64 = getattr(libdevice, fname64)
    impl_binary(key, float32, impl32)
    impl_binary(key, float64, impl64)
