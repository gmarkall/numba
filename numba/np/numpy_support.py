import collections
import ctypes
import re


from numba.core import errors, types, config
from numba.core.errors import TypingError

# re-export
from numba.core.cgutils import is_nonelike   # noqa: F401

re_typestr = re.compile(r'[<>=\|]([a-z])(\d+)?$', re.I)
re_datetimestr = re.compile(r'[<>=\|]([mM])8?(\[([a-z]+)\])?$', re.I)

sizeof_unicode_char = 4  # XXX: compiler-core: may be wrong on Windows


def _from_str_dtype(dtype):
    m = re_typestr.match(dtype.str)
    if not m:
        raise errors.NumbaNotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    if typecode == 'U':
        # unicode
        if dtype.byteorder not in '=|':
            raise errors.NumbaNotImplementedError("Does not support non-native "
                                                  "byteorder")
        count = dtype.itemsize // sizeof_unicode_char
        assert count == int(groups[1]), "Unicode char size mismatch"
        return types.UnicodeCharSeq(count)

    elif typecode == 'S':
        # char
        count = dtype.itemsize
        assert count == int(groups[1]), "Char size mismatch"
        return types.CharSeq(count)

    else:
        raise errors.NumbaNotImplementedError(dtype)


def _from_datetime_dtype(dtype):
    m = re_datetimestr.match(dtype.str)
    if not m:
        raise errors.NumbaNotImplementedError(dtype)
    groups = m.groups()
    typecode = groups[0]
    unit = groups[2] or ''
    if typecode == 'm':
        return types.NPTimedelta(unit)
    elif typecode == 'M':
        return types.NPDatetime(unit)
    else:
        raise errors.NumbaNotImplementedError(dtype)


def from_dtype(dtype):
    """
    Return a Numba Type instance corresponding to the given Numpy *dtype*.
    NumbaNotImplementedError is raised on unsupported Numpy dtypes.
    """
    if type(dtype) is type and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype)
    elif getattr(dtype, "fields", None) is not None:
        return from_struct_dtype(dtype)

    # XXX: compiler-core: initialize this once
    if config.USE_LEGACY_TYPE_SYSTEM:
        FROM_DTYPE = {
            np.dtype('bool'): types.boolean,
            np.dtype('int8'): types.int8,
            np.dtype('int16'): types.int16,
            np.dtype('int32'): types.int32,
            np.dtype('int64'): types.int64,

            np.dtype('uint8'): types.uint8,
            np.dtype('uint16'): types.uint16,
            np.dtype('uint32'): types.uint32,
            np.dtype('uint64'): types.uint64,

            np.dtype('float32'): types.float32,
            np.dtype('float64'): types.float64,
            np.dtype('float16'): types.float16,
            np.dtype('complex64'): types.complex64,
            np.dtype('complex128'): types.complex128,

            np.dtype(object): types.pyobject,
        }
    else:
        FROM_DTYPE = {
            np.dtype('bool'): types.np_bool_,
            np.dtype('int8'): types.np_int8,
            np.dtype('int16'): types.np_int16,
            np.dtype('int32'): types.np_int32,
            np.dtype('int64'): types.np_int64,

            np.dtype('uint8'): types.np_uint8,
            np.dtype('uint16'): types.np_uint16,
            np.dtype('uint32'): types.np_uint32,
            np.dtype('uint64'): types.np_uint64,

            np.dtype('float32'): types.np_float32,
            np.dtype('float64'): types.np_float64,
            np.dtype('float16'): types.np_float16,
            np.dtype('complex64'): types.np_complex64,
            np.dtype('complex128'): types.np_complex128,

            np.dtype(object): types.pyobject,
        }

    try:
        return FROM_DTYPE[dtype]
    except KeyError:
        pass

    try:
        char = dtype.char
    except AttributeError:
        pass
    else:
        if char in 'SU':
            return _from_str_dtype(dtype)
        if char in 'mM':
            return _from_datetime_dtype(dtype)
        if char in 'V' and dtype.subdtype is not None:
            subtype = from_dtype(dtype.subdtype[0])
            return types.NestedArray(subtype, dtype.shape)

    raise errors.NumbaNotImplementedError(dtype)


def as_dtype(nbtype):
    """
    Return a numpy dtype instance corresponding to the given Numba type.
    NotImplementedError is if no correspondence is known.
    """

    # XXX: compiler-core: initialize this once
    _as_dtype_letters = {
        types.NPDatetime: 'M8',
        types.NPTimedelta: 'm8',
        types.CharSeq: 'S',
        types.UnicodeCharSeq: 'U',
    }

    nbtype = types.unliteral(nbtype)
    if isinstance(nbtype, (types.Complex, types.Integer, types.Float)):
        return np.dtype(str(nbtype))
    if isinstance(nbtype, (types.Boolean)):
        return np.dtype('?')
    if isinstance(nbtype, (types.NPDatetime, types.NPTimedelta)):
        letter = _as_dtype_letters[type(nbtype)]
        if nbtype.unit:
            return np.dtype('%s[%s]' % (letter, nbtype.unit))
        else:
            return np.dtype(letter)
    if isinstance(nbtype, (types.CharSeq, types.UnicodeCharSeq)):
        letter = _as_dtype_letters[type(nbtype)]
        return np.dtype('%s%d' % (letter, nbtype.count))
    if isinstance(nbtype, types.Record):
        return as_struct_dtype(nbtype)
    if isinstance(nbtype, types.EnumMember):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.npytypes.DType):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.NumberClass):
        return as_dtype(nbtype.dtype)
    if isinstance(nbtype, types.NestedArray):
        spec = (as_dtype(nbtype.dtype), tuple(nbtype.shape))
        return np.dtype(spec)
    if isinstance(nbtype, types.PyObject):
        return np.dtype(object)

    msg = f"{nbtype} cannot be represented as a NumPy dtype"
    raise errors.NumbaNotImplementedError(msg)


def as_struct_dtype(rec):
    """Convert Numba Record type to NumPy structured dtype
    """
    assert isinstance(rec, types.Record)
    names = []
    formats = []
    offsets = []
    titles = []
    # Fill the fields if they are not a title.
    for k, t in rec.members:
        if not rec.is_title(k):
            names.append(k)
            formats.append(as_dtype(t))
            offsets.append(rec.offset(k))
            titles.append(rec.fields[k].title)

    fields = {
        'names': names,
        'formats': formats,
        'offsets': offsets,
        'itemsize': rec.size,
        'titles': titles,
    }
    _check_struct_alignment(rec, fields)
    return np.dtype(fields, align=rec.aligned)


def _check_struct_alignment(rec, fields):
    """Check alignment compatibility with Numpy"""
    if rec.aligned:
        for k, dt in zip(fields['names'], fields['formats']):
            llvm_align = rec.alignof(k)
            npy_align = dt.alignment
            if llvm_align is not None and npy_align != llvm_align:
                msg = (
                    'NumPy is using a different alignment ({}) '
                    'than Numba/LLVM ({}) for {}. '
                    'This is likely a NumPy bug.'
                )
                raise ValueError(msg.format(npy_align, llvm_align, dt))


def map_arrayscalar_type(val):
    if isinstance(val, np.generic):
        # We can't blindly call np.dtype() as it loses information
        # on some types, e.g. datetime64 and timedelta64.
        dtype = val.dtype
    else:
        try:
            dtype = np.dtype(type(val))
        except TypeError:
            raise errors.NumbaNotImplementedError("no corresponding numpy "
                                                  "dtype for %r" % type(val))
    return from_dtype(dtype)


def is_array(val):
    return isinstance(val, np.ndarray)


def map_layout(val):
    if val.flags['C_CONTIGUOUS']:
        layout = 'C'
    elif val.flags['F_CONTIGUOUS']:
        layout = 'F'
    else:
        layout = 'A'
    return layout


def select_array_wrapper(inputs):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    select the appropriate input for wrapping the operation output,
    according to each input's __array_priority__.

    An index into *inputs* is returned.
    """
    max_prio = float('-inf')
    selected_index = None
    for index, ty in enumerate(inputs):
        # Ties are broken by choosing the first winner, as in Numpy
        if (isinstance(ty, types.ArrayCompatible) and
                ty.array_priority > max_prio):
            selected_index = index
            max_prio = ty.array_priority

    assert selected_index is not None
    return selected_index


def resolve_output_type(context, inputs, formal_output):
    """
    Given the array-compatible input types to an operation (e.g. ufunc),
    and the operation's formal output type (a types.Array instance),
    resolve the actual output type using the typing *context*.

    This uses a mechanism compatible with Numpy's __array_priority__ /
    __array_wrap__.
    """
    selected_input = inputs[select_array_wrapper(inputs)]
    args = selected_input, formal_output
    sig = context.resolve_function_type('__array_wrap__', args, {})
    if sig is None:
        if selected_input.array_priority == types.Array.array_priority:
            # If it's the same priority as a regular array, assume we
            # should return the output unchanged.
            # (we can't define __array_wrap__ explicitly for types.Buffer,
            #  as that would be inherited by most array-compatible objects)
            return formal_output
        raise errors.TypingError("__array_wrap__ failed for %s" % (args,))
    return sig.return_type


def supported_ufunc_loop(ufunc, loop):
    """Return whether the *loop* for the *ufunc* is supported -in nopython-.

    *loop* should be a UFuncLoopSpec instance, and *ufunc* a numpy ufunc.

    For ufuncs implemented using the ufunc_db, it is supported if the ufunc_db
    contains a lowering definition for 'loop' in the 'ufunc' entry.

    For other ufuncs, it is type based. The loop will be considered valid if it
    only contains the following letter types: '?bBhHiIlLqQfd'. Note this is
    legacy and when implementing new ufuncs the ufunc_db should be preferred,
    as it allows for a more fine-grained incremental support.
    """
    # NOTE: Assuming ufunc for the CPUContext
    from numba.np import ufunc_db
    loop_sig = loop.ufunc_sig
    try:
        # check if the loop has a codegen description in the
        # ufunc_db. If so, we can proceed.

        # note that as of now not all ufuncs have an entry in the
        # ufunc_db
        supported_loop = loop_sig in ufunc_db.get_ufunc_info(ufunc)
    except KeyError:
        # for ufuncs not in ufunc_db, base the decision of whether the
        # loop is supported on its types
        loop_types = [x.char for x in loop.numpy_inputs + loop.numpy_outputs]
        supported_types = '?bBhHiIlLqQfd'
        # check if all the types involved in the ufunc loop are
        # supported in this mode
        supported_loop = all(t in supported_types for t in loop_types)

    return supported_loop


class UFuncLoopSpec(collections.namedtuple('_UFuncLoopSpec',
                                           ('inputs', 'outputs', 'ufunc_sig'))):
    """
    An object describing a ufunc loop's inner types.  Properties:
    - inputs: the inputs' Numba types
    - outputs: the outputs' Numba types
    - ufunc_sig: the string representing the ufunc's type signature, in
      Numpy format (e.g. "ii->i")
    """

    __slots__ = ()

    @property
    def numpy_inputs(self):
        return [as_dtype(x) for x in self.inputs]

    @property
    def numpy_outputs(self):
        return [as_dtype(x) for x in self.outputs]


def _is_aligned_struct(struct):
    return struct.isalignedstruct


def from_struct_dtype(dtype):
    """Convert a NumPy structured dtype to Numba Record type
    """
    if dtype.hasobject:
        msg = "dtypes that contain object are not supported."
        raise errors.NumbaNotImplementedError(msg)

    fields = []
    for name, info in dtype.fields.items():
        # *info* may have 3 element
        [elemdtype, offset] = info[:2]
        title = info[2] if len(info) == 3 else None

        ty = from_dtype(elemdtype)
        infos = {
            'type': ty,
            'offset': offset,
            'title': title,
        }
        fields.append((name, infos))

    # Note: dtype.alignment is not consistent.
    #       It is different after passing into a recarray.
    #       recarray(N, dtype=mydtype).dtype.alignment != mydtype.alignment
    size = dtype.itemsize
    aligned = _is_aligned_struct(dtype)

    return types.Record(fields, size, aligned)


def _get_bytes_buffer(ptr, nbytes):
    """
    Get a ctypes array of *nbytes* starting at *ptr*.
    """
    if isinstance(ptr, ctypes.c_void_p):
        ptr = ptr.value
    arrty = ctypes.c_byte * nbytes
    return arrty.from_address(ptr)


def _get_array_from_ptr(ptr, nbytes, dtype):
    return np.frombuffer(_get_bytes_buffer(ptr, nbytes), dtype)


def carray(ptr, shape, dtype=None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in C order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
    from numba.core.typing.ctypes_utils import from_ctypes

    try:
        # Use ctypes parameter protocol if available
        ptr = ptr._as_parameter_
    except AttributeError:
        pass

    # Normalize dtype, to accept e.g. "int64" or np.int64
    if dtype is not None:
        dtype = np.dtype(dtype)

    if isinstance(ptr, ctypes.c_void_p):
        if dtype is None:
            raise TypeError("explicit dtype required for void* argument")
        p = ptr
    elif isinstance(ptr, ctypes._Pointer):
        ptrty = from_ctypes(ptr.__class__)
        assert isinstance(ptrty, types.CPointer)
        ptr_dtype = as_dtype(ptrty.dtype)
        if dtype is not None and dtype != ptr_dtype:
            raise TypeError("mismatching dtype '%s' for pointer %s"
                            % (dtype, ptr))
        dtype = ptr_dtype
        p = ctypes.cast(ptr, ctypes.c_void_p)
    else:
        raise TypeError("expected a ctypes pointer, got %r" % (ptr,))

    nbytes = dtype.itemsize * np.prod(shape, dtype=np.intp)
    return _get_array_from_ptr(p, nbytes, dtype).reshape(shape)


def farray(ptr, shape, dtype=None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in Fortran order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
    if not isinstance(shape, int):
        shape = shape[::-1]
    return carray(ptr, shape, dtype).T


def is_contiguous(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of C layout?

    Note: The code is usable as a numba-compiled function
    """
    nd = len(dims)
    # Check and skip 1s or 0s in inner dims
    innerax = nd - 1
    while innerax > -1 and dims[innerax] <= 1:
        innerax -= 1

    # Early exit if all axis are 1s or 0s
    if innerax < 0:
        return True

    # Check itemsize matches innermost stride
    if itemsize != strides[innerax]:
        return False

    # Check and skip 1s or 0s in outer dims
    outerax = 0
    while outerax < innerax and dims[outerax] <= 1:
        outerax += 1

    # Check remaining strides to be contiguous
    ax = innerax
    while ax > outerax:
        if strides[ax] * dims[ax] != strides[ax - 1]:
            return False
        ax -= 1
    return True


def is_fortran(dims, strides, itemsize):
    """Is the given shape, strides, and itemsize of F layout?

    Note: The code is usable as a numba-compiled function
    """
    nd = len(dims)
    # Check and skip 1s or 0s in inner dims
    firstax = 0
    while firstax < nd and dims[firstax] <= 1:
        firstax += 1

    # Early exit if all axis are 1s or 0s
    if firstax >= nd:
        return True

    # Check itemsize matches innermost stride
    if itemsize != strides[firstax]:
        return False

    # Check and skip 1s or 0s in outer dims
    lastax = nd - 1
    while lastax > firstax and dims[lastax] <= 1:
        lastax -= 1

    # Check remaining strides to be contiguous
    ax = firstax
    while ax < lastax:
        if strides[ax] * dims[ax] != strides[ax + 1]:
            return False
        ax += 1
    return True


def type_can_asarray(arr):
    """ Returns True if the type of 'arr' is supported by the Numba `np.asarray`
    implementation, False otherwise.
    """

    ok = (types.Array, types.Sequence, types.Tuple, types.StringLiteral,
          types.Number, types.Boolean, types.containers.ListType)

    return isinstance(arr, ok)


def type_is_scalar(typ):
    """ Returns True if the type of 'typ' is a scalar type, according to
    NumPy rules. False otherwise.
    https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
    """

    ok = (types.Boolean, types.Number, types.UnicodeType, types.StringLiteral,
          types.NPTimedelta, types.NPDatetime)
    return isinstance(typ, ok)


def check_is_integer(v, name):
    """Raises TypingError if the value is not an integer."""
    if not isinstance(v, (int, types.Integer)):
        raise TypingError('{} must be an integer'.format(name))


def lt_floats(a, b):
    # Adapted from NumPy commit 717c7acf which introduced the behavior of
    # putting NaNs at the end.
    # The code is later moved to numpy/core/src/npysort/npysort_common.h
    # This info is gathered as of NumPy commit d8c09c50
    return a < b or (np.isnan(b) and not np.isnan(a))


def lt_complex(a, b):
    if np.isnan(a.real):
        if np.isnan(b.real):
            if np.isnan(a.imag):
                return False
            else:
                if np.isnan(b.imag):
                    return True
                else:
                    return a.imag < b.imag
        else:
            return False

    else:
        if np.isnan(b.real):
            return True
        else:
            if np.isnan(a.imag):
                if np.isnan(b.imag):
                    return a.real < b.real
                else:
                    return False
            else:
                if np.isnan(b.imag):
                    return True
                else:
                    if a.real < b.real:
                        return True
                    elif a.real == b.real:
                        return a.imag < b.imag
                    return False
