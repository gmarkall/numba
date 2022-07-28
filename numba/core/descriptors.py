"""
Target Descriptors
"""

import contextlib
from abc import ABCMeta, abstractmethod


class TargetDescriptor(metaclass=ABCMeta):

    def __init__(self, target_name):
        self._target_name = target_name

    @property
    @abstractmethod
    def typing_context(self):
        ...

    @property
    @abstractmethod
    def target_context(self):
        ...


class NestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        old_nested = self._typing_context, self._target_context
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            self._typing_context, self._target_context = old_nested


class NestableTargetDescriptor(TargetDescriptor):
    @property
    def typing_context(self):
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    @property
    def target_context(self):
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    def nested_context(self, typing_context, target_context):
        """
        A context manager temporarily replacing the contexts with the
        given ones, for the current thread of execution.
        """
        return self._nested.nested(typing_context, target_context)
