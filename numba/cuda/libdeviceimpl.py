from llvmlite.llvmpy.core import Type
from numba.core import types
from numba.core.imputils import Registry
from numba.cuda import libdevice, libdevicefuncs

registry = Registry()
lower = registry.lower


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


def libdevice_implement_tuple_return(func, retty, return_args, nb_retty,
                                     nbargs):
    def core(context, builder, sig, args):
        #from pudb import set_trace; set_trace()
        #lmod = builder.module
        #fretty = context.get_value_type(retty)
        fargtys = []
        for arg in nbargs:
            ty = context.get_value_type(arg.ty)
            if arg.is_ptr:
                ty = ty.as_pointer()
            fargtys.append
        print(f'{func}, {fargtys}')

    key = getattr(libdevice, func[5:])

    argtys = [arg.ty for arg in args if not arg.is_ptr]
    lower(key, *argtys)(core)


for func, (retty, args) in libdevicefuncs.functions.items():
    extra_return_argtys = [arg.ty for arg in args if arg.is_ptr]
    # __nv_nan[f] are weird
    if extra_return_argtys and func not in ('__nv_nan', '__nv_nanf'):
        if retty != types.void:
            return_argtys = [retty] + extra_return_argtys
        else:
            return_argtys = extra_return_argtys
        print(f'{func}, {return_argtys}')
        nb_retty = types.Tuple(return_argtys)
        libdevice_implement_tuple_return(func, retty, return_argtys, nb_retty,
                                         args)

    libdevice_implement(func, retty, args)
