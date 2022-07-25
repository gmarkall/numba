import numpy as np


class UseCase:
    """
    Provide a way to call a kernel as if it were a function.

    This allows the CUDA cache tests to closely match the CPU cache tests, and
    also to support calling cache use cases as njitted functions. The class
    wraps a function that takes an array for the return value and arguments,
    and provides an interface that accepts arguments, launches the kernel
    appropriately, and returns the stored return value.

    The return type is inferred from the type of the first argument, unless it
    is explicitly overridden by the ``retty`` kwarg.
    """
    def __init__(self, func, retty=None):
        self._func = func
        self._retty = retty

    def __call__(self, *args):
        array_args = [np.asarray(arg) for arg in args]
        if self._retty:
            array_return = np.ndarray((), dtype=self._retty)
        else:
            array_return = np.zeros_like(array_args[0])

        self._call(array_return, *array_args)
        return array_return[()]

    @property
    def func(self):
        return self._func


class CUDAUseCase(UseCase):
    def _call(self, ret, *args):
        self._func[1, 1](ret, *args)


class CPUUseCase(UseCase):
    def _call(self, ret, *args):
        self._func(ret, *args)
