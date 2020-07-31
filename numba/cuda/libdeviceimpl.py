import math
from llvmlite.llvmpy.core import Type
from numba.core import types, cgutils
from numba.core.imputils import Registry
from numba.cuda import libdevice, libdevicefuncs

registry = Registry()
lower = registry.lower


def powi_implement(nvname):
    def core(context, builder, sig, args):
        [base, pow] = args
        [basety, powty] = sig.args
        lmod = builder.module
        fty = context.get_value_type(basety)
        ity = context.get_value_type(types.int32)
        fnty = Type.function(fty, [fty, ity])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        return builder.call(fn, [base, pow])

    return core


lower(math.pow, types.float32, types.int32)(powi_implement('__nv_powif'))
lower(math.pow, types.float64, types.int32)(powi_implement('__nv_powi'))


def modf_implement(nvname, ty):
    def core(context, builder, sig, args):
        arg, = args
        argty, = sig.args
        fty = context.get_value_type(argty)
        lmod = builder.module
        ptr = cgutils.alloca_once(builder, fty)
        fnty = Type.function(fty, [fty, fty.as_pointer()])
        fn = lmod.get_or_insert_function(fnty, name=nvname)
        out = builder.call(fn, [arg, ptr])
        ret = context.make_tuple(builder, types.UniTuple(argty, 2),
                                 [out, builder.load(ptr)])
        return ret
    return core


for (ty, intrin) in ((types.float64, '__nv_modf',),
                     (types.float32, '__nv_modff',)):
    lower(math.modf, ty)(modf_implement(intrin, ty))


def libdevice_implement(func, retty, nbargs):
    def core(context, builder, sig, args):
        lmod = builder.module
        fretty = context.get_value_type(retty)
        fargtys = [context.get_value_type(arg.ty) for arg in nbargs if not
                   arg.is_ptr]
        fnty = Type.function(fretty, fargtys)
        fn = lmod.get_or_insert_function(fnty, name=func)
        return builder.call(fn, args)

    key = getattr(libdevice, func[5:])

    argtys = [arg.ty for arg in args if not arg.is_ptr]
    lower(key, *argtys)(core)


for func, (retty, args) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        print(f'Skipping {func} with pointer arg')
        continue
    libdevice_implement(func, retty, args)
