import math
from llvmlite import ir
from numba.core import types, typing
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice

registry = Registry()
lower = registry.lower


booleans = []
booleans += [('isnand', 'isnanf', math.isnan)]
booleans += [('isinfd', 'isinff', math.isinf)]

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


@lower(math.isinf, types.Integer)
@lower(math.isnan, types.Integer)
def math_isinf_isnan_int(context, builder, sig, args):
    return context.get_constant(types.boolean, 0)


def impl_boolean(key, ty, libfunc):
    def lower_boolean_impl(context, builder, sig, args):
        libfunc_impl = context.get_function(libfunc,
                                            typing.signature(types.int32, ty))
        result = libfunc_impl(builder, args)
        return context.cast(builder, result, types.int32, types.boolean)

    lower(key, ty)(lower_boolean_impl)


def impl_unary(key, ty, libfunc):
    def lower_unary_impl(context, builder, sig, args):
        libfunc_impl = context.get_function(libfunc, typing.signature(ty, ty))
        return libfunc_impl(builder, args)

    lower(key, ty)(lower_unary_impl)


def impl_unary_int(key, ty, libfunc):
    def lower_unary_int_impl(context, builder, sig, args):
        if sig.args[0] == int64:
            convert = builder.sitofp
        else:
            convert = builder.uitofp
        arg = convert(args[0], ir.DoubleType())
        sig = typing.signature(float64, float64)
        libfunc_impl = context.get_function(libfunc, sig)
        return libfunc_impl(builder, [arg])

    lower(key, ty)(lower_unary_int_impl)


def impl_binary(key, ty, libfunc):
    def lower_binary_impl(context, builder, sig, args):
        libfunc_impl = context.get_function(libfunc,
                                            typing.signature(ty, ty, ty))
        return libfunc_impl(builder, args)

    lower(key, ty, ty)(lower_binary_impl)


def impl_binary_int(key, ty, libfunc):
    def lower_binary_int_impl(context, builder, sig, args):
        if sig.args[0] == int64:
            convert = builder.sitofp
        else:
            convert = builder.uitofp
        args = [convert(arg, ir.DoubleType()) for arg in args]
        sig = typing.signature(float64, float64, float64)
        libfunc_impl = context.get_function(libfunc, sig)
        return libfunc_impl(builder, args)

    lower(key, ty, ty)(lower_binary_int_impl)


for fname64, fname32, key in booleans:
    impl32 = getattr(libdevice, fname32)
    impl64 = getattr(libdevice, fname64)
    impl_boolean(key, float32, impl32)
    impl_boolean(key, float64, impl64)


for fname64, fname32, key in unarys:
    impl32 = getattr(libdevice, fname32)
    impl64 = getattr(libdevice, fname64)
    impl_unary(key, float32, impl32)
    impl_unary(key, float64, impl64)
    impl_unary_int(key, int64, impl64)
    impl_unary_int(key, uint64, impl64)


for fname64, fname32, key in binarys:
    impl32 = getattr(libdevice, fname32)
    impl64 = getattr(libdevice, fname64)
    impl_binary(key, float32, impl32)
    impl_binary(key, float64, impl64)
    impl_binary_int(key, int64, impl64)
    impl_binary_int(key, uint64, impl64)


def impl_pow(ty, libfunc):
    def lower_pow_impl(context, builder, sig, args):
        pow_sig = typing.signature(ty, ty, ty)
        libfunc_impl = context.get_function(libfunc, pow_sig)
        return libfunc_impl(builder, args)

    lower(math.pow, ty, types.int32)(lower_pow_impl)


def impl_pow_int(ty, libfunc):
    def lower_pow_impl_int(context, builder, sig, args):
        powi_sig = typing.signature(ty, ty, types.int32)
        libfunc_impl = context.get_function(libfunc, powi_sig)
        return libfunc_impl(builder, args)

    lower(math.pow, ty, types.int32)(lower_pow_impl_int)


impl_pow(types.float32, libdevice.powf)
impl_pow(types.float64, libdevice.pow)
impl_pow_int(types.float32, libdevice.powif)
impl_pow_int(types.float64, libdevice.powi)


def impl_modf(ty, libfunc):
    retty = types.Tuple((ty, ty))

    def lower_modf_impl(context, builder, sig, args):
        modf_sig = typing.signature(retty, ty)
        libfunc_impl = context.get_function(libfunc, modf_sig)
        return libfunc_impl(builder, args)

    lower(math.modf, ty)(lower_modf_impl)


impl_modf(types.float32, libdevice.modff)
impl_modf(types.float64, libdevice.modf)
