from llvmlite.llvmpy.core import Type
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


def libdevice_implement_multiple_returns(func, retty, nbargs):
    print(f'Skipping lowering of {func} with pointer arg')


for func, (retty, args) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        libdevice_implement_multiple_returns(func, retty, args)
    else:
        libdevice_implement(func, retty, args)
