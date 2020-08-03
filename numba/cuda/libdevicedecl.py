from numba.cuda import libdevice, libdevicefuncs
from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, Registry, signature

registry = Registry()
register_global = registry.register_global


def libdevice_declare(func, retty, args):
    class Libdevice_function(ConcreteTemplate):
        cases = [signature(retty, *[arg.ty for arg in args])]

    pyfunc = getattr(libdevice, func[5:])
    register_global(pyfunc)(Libdevice_function)


def libdevice_declare_multiple_returns(func, retty, args):
    # Any pointer arguments should be part of the return type.
    return_types = [arg.ty for arg in args if arg.is_ptr]
    # If the return type is void, there is no point adding it to the list of
    # return types.
    if retty != types.void:
        return_types.insert(0, retty)

    retty = types.Tuple(return_types)
    argtypes = [arg.ty for arg in args if not arg.is_ptr]

    class Libdevice_function(ConcreteTemplate):
        cases = [signature(retty, *argtypes)]

    pyfunc = getattr(libdevice, func[5:])
    register_global(pyfunc)(Libdevice_function)


for func, (retty, args) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        libdevice_declare_multiple_returns(func, retty, args)
    libdevice_declare(func, retty, args)
