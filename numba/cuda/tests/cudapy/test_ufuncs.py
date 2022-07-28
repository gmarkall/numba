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

    def test_sin_ufunc(self):
        self.basic_ufunc_test(np.sin, flags=no_pyobj_flags, kinds='cf')


if __name__ == '__main__':
    unittest.main()
