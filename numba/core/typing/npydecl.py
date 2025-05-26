import warnings

import operator

from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
                                         CallableTemplate, Registry, signature)

from numba.np.numpy_support import (as_dtype, from_dtype, resolve_output_type,
                                    carray, farray)
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
                               NumbaTypeError, NumbaAssertionError)

from numba.np_internal import np_nditer

registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


# -----------------------------------------------------------------------------
# Install global helpers for array methods.

class Numpy_method_redirection(AbstractTemplate):
    """
    A template redirecting a Numpy global function (e.g. np.sum) to an
    array method of the same name (e.g. ndarray.sum).
    """

    # Arguments like *axis* can specialize on literals but also support
    # non-literals
    prefer_literal = True

    def generic(self, args, kws):
        pysig = None
        if kws:
            if self.method_name == 'sum':
                if 'axis' in kws and 'dtype' not in kws:
                    def sum_stub(arr, axis):
                        pass
                    pysig = utils.pysignature(sum_stub)
                elif 'dtype' in kws and 'axis' not in kws:
                    def sum_stub(arr, dtype):
                        pass
                    pysig = utils.pysignature(sum_stub)
                elif 'dtype' in kws and 'axis' in kws:
                    def sum_stub(arr, axis, dtype):
                        pass
                    pysig = utils.pysignature(sum_stub)
            elif self.method_name == 'argsort':
                def argsort_stub(arr, kind='quicksort'):
                    pass
                pysig = utils.pysignature(argsort_stub)
            else:
                fmt = "numba doesn't support kwarg for {}"
                raise TypingError(fmt.format(self.method_name))

        arr = args[0]
        # This will return a BoundFunction
        meth_ty = self.context.resolve_getattr(arr, self.method_name)
        # Resolve arguments on the bound function
        meth_sig = self.context.resolve_function_type(meth_ty, args[1:], kws)
        if meth_sig is not None:
            return meth_sig.as_function().replace(pysig=pysig)


# -----------------------------------------------------------------------------
# Numpy array constructors

def parse_shape(shape):
    """
    Given a shape, return the number of dimensions.
    """
    ndim = None
    if isinstance(shape, types.Integer):
        ndim = 1
    elif isinstance(shape, (types.Tuple, types.UniTuple)):
        int_tys = (types.Integer, types.IntEnumMember)
        if all(isinstance(s, int_tys) for s in shape):
            ndim = len(shape)
    return ndim

def parse_dtype(dtype):
    """
    Return the dtype of a type, if it is either a DtypeSpec (used for most
    dtypes) or a TypeRef (used for record types).
    """
    if isinstance(dtype, types.DTypeSpec):
        return dtype.dtype
    elif isinstance(dtype, types.TypeRef):
        return dtype.instance_type
    elif isinstance(dtype, types.StringLiteral):
        dtstr = dtype.literal_value
        try:
            dt = np.dtype(dtstr)
        except TypeError:
            msg = f"Invalid NumPy dtype specified: '{dtstr}'"
            raise TypingError(msg)
        return from_dtype(dt)

def _parse_nested_sequence(context, typ):
    """
    Parse a (possibly 0d) nested sequence type.
    A (ndim, dtype) tuple is returned.  Note the sequence may still be
    heterogeneous, as long as it converts to the given dtype.
    """
    if isinstance(typ, (types.Buffer,)):
        raise TypingError("%s not allowed in a homogeneous sequence" % typ)
    elif isinstance(typ, (types.Sequence,)):
        n, dtype = _parse_nested_sequence(context, typ.dtype)
        return n + 1, dtype
    elif isinstance(typ, (types.BaseTuple,)):
        if typ.count == 0:
            # Mimic Numpy's behaviour
            return 1, types.float64
        n, dtype = _parse_nested_sequence(context, typ[0])
        dtypes = [dtype]
        for i in range(1, typ.count):
            _n, dtype = _parse_nested_sequence(context, typ[i])
            if _n != n:
                raise TypingError("type %s does not have a regular shape"
                                  % (typ,))
            dtypes.append(dtype)
        dtype = context.unify_types(*dtypes)
        if dtype is None:
            raise TypingError("cannot convert %s to a homogeneous type" % typ)
        return n + 1, dtype
    else:
        # Scalar type => check it's valid as a Numpy array dtype
        as_dtype(typ)
        return 0, typ


def _infer_dtype_from_inputs(inputs):
    return dtype


def _homogeneous_dims(context, func_name, arrays):
    ndim = arrays[0].ndim
    for a in arrays:
        if a.ndim != ndim:
            msg = (f"{func_name}(): all the input arrays must have same number "
                   "of dimensions")
            raise NumbaTypeError(msg)
    return ndim

def _sequence_of_arrays(context, func_name, arrays,
                        dim_chooser=_homogeneous_dims):
    if (not isinstance(arrays, types.BaseTuple)
        or not len(arrays)
        or not all(isinstance(a, types.Array) for a in arrays)):
        raise TypingError("%s(): expecting a non-empty tuple of arrays, "
                          "got %s" % (func_name, arrays))

    ndim = dim_chooser(context, func_name, arrays)

    dtype = context.unify_types(*(a.dtype for a in arrays))
    if dtype is None:
        raise TypingError("%s(): input arrays must have "
                          "compatible dtypes" % func_name)

    return dtype, ndim

def _choose_concatenation_layout(arrays):
    # Only create a F array if all input arrays have F layout.
    # This is a simplified version of Numpy's behaviour,
    # while Numpy's actually processes the input strides to
    # decide on optimal output strides
    # (see PyArray_CreateMultiSortedStridePerm()).
    return 'F' if all(a.layout == 'F' for a in arrays) else 'C'


# -----------------------------------------------------------------------------
# Linear algebra


class MatMulTyperMixin(object):

    def matmul_typer(self, a, b, out=None):
        """
        Typer function for Numpy matrix multiplication.
        """
        if not isinstance(a, types.Array) or not isinstance(b, types.Array):
            return
        if not all(x.ndim in (1, 2) for x in (a, b)):
            raise TypingError("%s only supported on 1-D and 2-D arrays"
                              % (self.func_name, ))
        # Output dimensionality
        ndims = set([a.ndim, b.ndim])
        if ndims == set([2]):
            # M * M
            out_ndim = 2
        elif ndims == set([1, 2]):
            # M* V and V * M
            out_ndim = 1
        elif ndims == set([1]):
            # V * V
            out_ndim = 0

        if out is not None:
            if out_ndim == 0:
                raise TypingError(
                    "explicit output unsupported for vector * vector")
            elif out.ndim != out_ndim:
                raise TypingError(
                    "explicit output has incorrect dimensionality")
            if not isinstance(out, types.Array) or out.layout != 'C':
                raise TypingError("output must be a C-contiguous array")
            all_args = (a, b, out)
        else:
            all_args = (a, b)

        if not (config.DISABLE_PERFORMANCE_WARNINGS or
                all(x.layout in 'CF' for x in (a, b))):
            msg = ("%s is faster on contiguous arrays, called on %s" %
                   (self.func_name, (a, b)))
            warnings.warn(NumbaPerformanceWarning(msg))
        if not all(x.dtype == a.dtype for x in all_args):
            raise TypingError("%s arguments must all have "
                              "the same dtype" % (self.func_name,))
        if not isinstance(a.dtype, (types.Float, types.Complex)):
            raise TypingError("%s only supported on "
                              "float and complex arrays"
                              % (self.func_name,))
        if out:
            return out
        elif out_ndim > 0:
            return types.Array(a.dtype, out_ndim, 'C')
        else:
            return a.dtype


def _check_linalg_matrix(a, func_name):
    if not isinstance(a, types.Array):
        return
    if not a.ndim == 2:
        raise TypingError("np.linalg.%s() only supported on 2-D arrays"
                          % func_name)
    if not isinstance(a.dtype, (types.Float, types.Complex)):
        raise TypingError("np.linalg.%s() only supported on "
                          "float and complex arrays" % func_name)

# -----------------------------------------------------------------------------
# Miscellaneous functions


@infer_global(np_nditer)
class NdIter(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) != 1:
            return
        arrays, = args

        if isinstance(arrays, types.BaseTuple):
            if not arrays:
                return
            arrays = list(arrays)
        else:
            arrays = [arrays]
        nditerty = types.NumpyNdIterType(arrays)
        return signature(nditerty, *args)


@infer_global(operator.eq)
class DtypeEq(AbstractTemplate):
    def generic(self, args, kws):
        [lhs, rhs] = args
        if isinstance(lhs, types.DType) and isinstance(rhs, types.DType):
            return signature(types.boolean, lhs, rhs)
