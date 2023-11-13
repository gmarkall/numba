"""
Hints to wrap Kernel arguments to indicate how to manage host-device
memory transfers before & after the kernel call.
"""
import abc
import ctypes
import numpy as np

from numba import types
from numba.core.typing.typeof import typeof, Purpose
from functools import singledispatch


class ArgHint(metaclass=abc.ABCMeta):
    def __init__(self, value):
        self.value = value

    @abc.abstractmethod
    def to_device(self, retr, stream=0):
        """
        :param stream: a stream to use when copying data
        :param retr:
            a list of clean-up work to do after the kernel's been run.
            Append 0-arg lambdas to it!
        :return: a value (usually an `DeviceNDArray`) to be passed to
            the kernel
        """
        pass

    @property
    def _numba_type_(self):
        return typeof(self.value, Purpose.argument)


class In(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, _ = auto_device(
            self.value,
            stream=stream)
        # A dummy writeback functor to keep devary alive until the kernel
        # is called.
        retr.append(lambda: devary)
        return devary


class Out(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, conv = auto_device(
            self.value,
            copy=False,
            stream=stream)
        if conv:
            retr.append(lambda: devary.copy_to_host(self.value, stream=stream))
        return devary


class InOut(ArgHint):
    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, conv = auto_device(
            self.value,
            stream=stream)
        if conv:
            retr.append(lambda: devary.copy_to_host(self.value, stream=stream))
        return devary


def wrap_arg(value, default=InOut):
    return value if isinstance(value, ArgHint) else default(value)


@singledispatch
def prepare_arg(ty, val):
    raise NotImplementedError(f'Unsupported argument type {ty}')


@prepare_arg.register(types.Integer)
def prepare_int_arg(ty, val):
    return getattr(ctypes, "c_%s" % ty)(val)


@prepare_arg.register(types.float16)
def prepare_float16_arg(ty, val):
    return ctypes.c_uint16(np.float16(val).view(np.uint16))


@prepare_arg.register(types.float32)
def prepare_float32_arg(ty, val):
    return ctypes.c_float(val)


@prepare_arg.register(types.float64)
def prepare_float64_arg(ty, val):
    return ctypes.c_double(val)


@prepare_arg.register(types.boolean)
def prepare_boolean_arg(ty, val):
    return ctypes.c_uint8(int(val))


@prepare_arg.register(types.complex64)
def prepare_complex64_arg(ty, val):
    c_real = ctypes.c_float(val.real)
    c_imag = ctypes.c_float(val.imag)
    return c_real, c_imag


@prepare_arg.register(types.complex128)
def prepare_complex128_arg(ty, val):
    c_real = ctypes.c_double(val.real)
    c_imag = ctypes.c_double(val.imag)
    return c_real, c_imag


@prepare_arg.register(types.NPDatetime)
@prepare_arg.register(types.NPTimedelta)
def prepare_datetime_arg(ty, val):
    return ctypes.c_int64(val.view(np.int64))


__all__ = [
    'In',
    'Out',
    'InOut',

    'ArgHint',
    'wrap_arg',
]
