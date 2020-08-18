from llvmlite.llvmpy.core import Type
from numba.core import cgutils, types
from numba.core.imputils import Registry
from numba.cuda import libdevice, libdevicefuncs

registry = Registry()
lower = registry.lower


def libdevice_implement(func, retty, nbargs):
    def core(context, builder, sig, args):
        lmod = builder.module
        fretty = context.get_value_type(retty)
        fargtys = [context.get_value_type(arg.ty) for arg in nbargs]
        fnty = Type.function(fretty, fargtys)
        fn = lmod.get_or_insert_function(fnty, name=func)
        return builder.call(fn, args)

    key = getattr(libdevice, func[5:])

    argtys = [arg.ty for arg in args if not arg.is_ptr]
    lower(key, *argtys)(core)


def libdevice_implement_multiple_returns(func, retty, prototype_args):
    # Any pointer arguments should be part of the return type.
    nb_return_types = [arg.ty for arg in prototype_args if arg.is_ptr]
    # If the return type is void, there is no point adding it to the list of
    # return types.
    if retty != types.void:
        nb_return_types.insert(0, retty)

    if len(nb_return_types) > 1:
        nb_retty = types.Tuple(nb_return_types)
    else:
        nb_retty = nb_return_types[0]

    nb_argtypes = [arg.ty for arg in prototype_args if not arg.is_ptr]

    def core(context, builder, sig, args):
        lmod = builder.module

        fargtys = []
        for arg in prototype_args:
            ty = context.get_value_type(arg.ty)
            if arg.is_ptr:
                ty = ty.as_pointer()
            fargtys.append(ty)

        fretty = context.get_value_type(retty)

        fnty = Type.function(fretty, fargtys)
        fn = lmod.get_or_insert_function(fnty, name=func)

        actual_args = []
        virtual_args = []
        arg_idx = 0
        for arg in prototype_args:
            if arg.is_ptr:
                # Allocate virtual arg and add to args
                tmp_arg = cgutils.alloca_once(builder,
                                              context.get_value_type(arg.ty))
                actual_args.append(tmp_arg)
                virtual_args.append(tmp_arg)
            else:
                actual_args.append(args[arg_idx])
                arg_idx += 1

        ret = builder.call(fn, actual_args)

        tuple_args = []
        if retty != types.void:
            tuple_args.append(ret)
        for arg in virtual_args:
            tuple_args.append(builder.load(arg))

        if isinstance(nb_retty, types.UniTuple):
            return cgutils.pack_array(builder, tuple_args)
        else:
            return cgutils.pack_struct(builder, tuple_args)

    key = getattr(libdevice, func[5:])
    lower(key, *nb_argtypes)(core)


for func, (retty, args) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        libdevice_implement_multiple_returns(func, retty, args)
    else:
        libdevice_implement(func, retty, args)
