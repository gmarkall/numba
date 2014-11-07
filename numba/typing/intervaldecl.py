from numba.typing.templates import AttributeTemplate, Registry
from numba.types import float64, bool_, interval_type

registry = Registry()
builtin_attr = registry.register_attr

@builtin_attr
class IntervalAttributes(AttributeTemplate):
    key = interval_type

    # We will store the interval bounds as 64-bit floats
    _attributes = dict(lo=float64, hi=float64)

    def generic_resolve(self, value, attr):
        return self._attributes[attr]
