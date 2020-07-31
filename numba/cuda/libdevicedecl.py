from numba.cuda import libdevice, libdevicefuncs

from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, Registry, signature

registry = Registry()
register_global = registry.register_global


def libdevice_declare(func, retty, argtys):
    class Libdevice_function(ConcreteTemplate):
        cases = [signature(retty, *argtys)]

    pyfunc = getattr(libdevice, func[5:])
    register_global(pyfunc)(Libdevice_function)


for func, (retty, args) in libdevicefuncs.functions.items():
    extra_return_argtys = [arg.ty for arg in args if arg.is_ptr]
    # We strip off the pointer arguments from the signature, because they
    # will be return values in CUDA Python.
    argtys = [arg.ty for arg in args if not arg.is_ptr]
    if extra_return_argtys:
        if retty != types.void:
            return_argtys = [retty] + extra_return_argtys
        else:
            return_argtys = extra_return_argtys
        print(f'{func}, {return_argtys}')
        retty = types.Tuple(return_argtys)

    libdevice_declare(func, retty, argtys)
