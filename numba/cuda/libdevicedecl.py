from numba.cuda import libdevice, libdevicefuncs

from numba.core.typing.templates import ConcreteTemplate, Registry, signature

registry = Registry()
register_global = registry.register_global


def libdevice_declare(func, retty, args):
    class Libdevice_function(ConcreteTemplate):
        cases = [signature(retty, *[arg.ty for arg in args])]

    pyfunc = getattr(libdevice, func[5:])
    register_global(pyfunc)(Libdevice_function)


def libdevice_declare_multiple_returns(func, retty, args):
    #class Libdevice_function(ConcreteTemplate):
    #    cases = [signature(retty, *[arg.ty for arg in args if not arg.is_ptr])]

    #pyfunc = getattr(libdevice, func[5:])
    #register_global(pyfunc)(Libdevice_function)
    print(f"Skipping typing {func} with pointer arg")


for func, (retty, args) in libdevicefuncs.functions.items():
    if any([arg.is_ptr for arg in args]):
        libdevice_declare_multiple_returns(func, retty, args)
    libdevice_declare(func, retty, args)
