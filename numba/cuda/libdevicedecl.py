from numba.cuda import libdevice, libdevicefuncs

from numba.core.typing.templates import ConcreteTemplate, Registry, signature

registry = Registry()
register_global = registry.register_global


def libdevice_declare(func, retty, args):
    class Libdevice_function(ConcreteTemplate):
        cases = [signature(retty, *[arg.ty for arg in args if not arg.is_ptr])]

    pyfunc = getattr(libdevice, func[5:])
    register_global(pyfunc)(Libdevice_function)


for func, (retty, args) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        print(f'Skipping {func} with pointer arg')
        continue
    libdevice_declare(func, retty, args)
