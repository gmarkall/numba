import numpy as np
import warnings

from numba import cuda, types
from numba.cuda.descriptor import cuda_target
from numba.cuda.testing import CUDATestCase, unittest
from numba.tests.support import CompilationCache
from numba.tests.test_ufuncs import BaseUFuncTest, no_pyobj_flags


def _make_ufunc_usecase(ufunc):
    ldict = {}
    arg_str = ','.join(['a{0}'.format(i) for i in range(ufunc.nargs)])
    func_str = f'def fn({arg_str}):\n    np.{ufunc.__name__}({arg_str})'
    print(func_str)
    exec(func_str, globals(), ldict)
    fn = ldict['fn']
    fn.__name__ = '{0}_usecase'.format(ufunc.__name__)
    return fn


class TestUFuncs(BaseUFuncTest, CUDATestCase):
    def setUp(self):
        CUDATestCase.setUp(self)
        BaseUFuncTest.setUp(self)

        # The base ufunc test does not run with complex inputs
        self.inputs.extend([
            (np.complex64(-0.5 - 0.5j), types.complex64),
            (np.complex64(0.0), types.complex64),
            (np.complex64(0.5 + 0.5j), types.complex64),

            (np.complex64(-0.5 - 0.5j), types.complex128),
            (np.complex64(0.0), types.complex128),
            (np.complex64(0.5 + 0.5j), types.complex128),

            (np.array([-0.5 - 0.5j, 0.0, 0.5 + 0.5j], dtype='c8'),
             types.Array(types.complex64, 1, 'C')),
            (np.array([-0.5 - 0.5j, 0.0, 0.5 + 0.5j], dtype='c16'),
             types.Array(types.complex128, 1, 'C')),
        ])

    # Copied from TestUFuncs in the main tests
    def basic_ufunc_test(self, ufunc, flags=no_pyobj_flags,
                         skip_inputs=[], additional_inputs=[],
                         int_output_type=None, float_output_type=None,
                         kinds='ifc', positive_only=False):

        # Necessary to avoid some Numpy warnings being silenced, despite
        # the simplefilter() call below.
        self.reset_module_warnings(__name__)

        pyfunc = _make_ufunc_usecase(ufunc)

        inputs = list(self.inputs) + additional_inputs

        for input_tuple in inputs:
            input_operand = input_tuple[0]
            input_type = input_tuple[1]

            is_tuple = isinstance(input_operand, tuple)
            if is_tuple:
                args = input_operand
            else:
                args = (input_operand,) * ufunc.nin

            if input_type in skip_inputs:
                continue
            if positive_only and np.any(args[0] < 0):
                continue

            # Some ufuncs don't allow all kinds of arguments
            if (args[0].dtype.kind not in kinds):
                continue

            output_type = self._determine_output_type(
                input_type, int_output_type, float_output_type)

            input_types = (input_type,) * ufunc.nin
            output_types = (output_type,) * ufunc.nout
            #cr = self.cache.compile(pyfunc, input_types + output_types,
            #                        flags=flags)
            #cfunc = cr.entry_point
            argtypes = input_types + output_types
            cfunc = cuda.jit(argtypes)(pyfunc)

            if isinstance(args[0], np.ndarray):
                results = [
                    np.zeros(args[0].size,
                             dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]
                expected = [
                    np.zeros(args[0].size, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]
            else:
                results = [
                    np.zeros(1, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]
                expected = [
                    np.zeros(1, dtype=out_ty.dtype.name)
                    for out_ty in output_types
                ]

            invalid_flag = False
            with warnings.catch_warnings(record=True) as warnlist:
                warnings.simplefilter('always')
                pyfunc(*args, *expected)

                warnmsg = "invalid value encountered"
                for thiswarn in warnlist:

                    if (issubclass(thiswarn.category, RuntimeWarning)
                            and str(thiswarn.message).startswith(warnmsg)):
                        invalid_flag = True

            print(f"Running with {args} {results}")
            cfunc[1, 1](*args, *results)
            print(f"Got {results}")

            for expected_i, result_i in zip(expected, results):
                msg = '\n'.join(["ufunc '{0}' failed",
                                 "inputs ({1}):", "{2}",
                                 "got({3})", "{4}",
                                 "expected ({5}):", "{6}"
                                 ]).format(ufunc.__name__,
                                           input_type, input_operand,
                                           output_type, result_i,
                                           expected_i.dtype, expected_i)
                try:
                    np.testing.assert_array_almost_equal(
                        expected_i, result_i,
                        decimal=5,
                        err_msg=msg)
                except AssertionError:
                    if invalid_flag:
                        # Allow output to mismatch for invalid input
                        print("Output mismatch for invalid input",
                              input_tuple, result_i, expected_i)
                    else:
                        raise

    def _init_cache(self):
        self.cache = CompilationCache(cuda_target)

    ############################################################################
    # Trigonometric Functions

    def test_sin_ufunc(self):
        self.basic_ufunc_test(np.sin, flags=no_pyobj_flags, kinds='cf')

    def test_cos_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.cos, flags=flags, kinds='cf')

    def test_tan_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.tan, flags=flags, kinds='cf')

    def test_arcsin_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.arcsin, flags=flags, kinds='cf')

    def test_arccos_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.arccos, flags=flags, kinds='cf')

    def test_arctan_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.arctan, flags=flags, kinds='cf')

    def test_arctan2_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.arctan2, flags=flags, kinds='cf')

    def test_hypot_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.hypot, kinds='f')

    def test_sinh_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.sinh, flags=flags, kinds='cf')

    def test_cosh_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.cosh, flags=flags, kinds='cf')

    def test_tanh_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.tanh, flags=flags, kinds='cf')

    def test_arcsinh_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.arcsinh, flags=flags, kinds='cf')

    def test_arccosh_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.arccosh, flags=flags, kinds='cf')

    def test_arctanh_ufunc(self, flags=no_pyobj_flags):
        # arctanh is only valid is only finite in the range ]-1, 1[
        # This means that for any of the integer types it will produce
        # conversion from infinity/-infinity to integer. That's undefined
        # behavior in C, so the results may vary from implementation to
        # implementation. This means that the result from the compiler
        # used to compile NumPy may differ from the result generated by
        # llvm. Skipping the integer types in this test avoids failed
        # tests because of this.
        to_skip = [types.Array(types.uint32, 1, 'C'), types.uint32,
                   types.Array(types.int32, 1, 'C'), types.int32,
                   types.Array(types.uint64, 1, 'C'), types.uint64,
                   types.Array(types.int64, 1, 'C'), types.int64]

        self.basic_ufunc_test(np.arctanh, skip_inputs=to_skip, flags=flags,
                              kinds='cf')

    def test_deg2rad_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.deg2rad, flags=flags, kinds='f')

    def test_rad2deg_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.rad2deg, flags=flags, kinds='f')

    def test_degrees_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.degrees, flags=flags, kinds='f')

    def test_radians_ufunc(self, flags=no_pyobj_flags):
        self.basic_ufunc_test(np.radians, flags=flags, kinds='f')


if __name__ == '__main__':
    unittest.main()
