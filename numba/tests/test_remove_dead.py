#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

import numba
from numba import njit, jit
from numba.core import ir_utils
from numba.core import types, ir,  compiler
from numba.core.registry import cpu_target
from numba.core.ir_utils import (copy_propagate, apply_copy_propagate,
                            get_name_var_table, remove_dels, remove_dead,
                            remove_call_handlers, alias_func_extensions)
from numba.core.typed_passes import type_inference_stage
from numba.core.compiler_machinery import FunctionPass, register_pass, PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                             IRProcessing, DeadBranchPrune,
                             RewriteSemanticConstants, GenericRewrites,
                             WithLifting, PreserveIR, InlineClosureLikes)

from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
                                     NopythonRewrites, NativeLowering,
                                     IRLegalization, NoPythonBackend)
import numpy as np
from numba.tests.support import needs_blas, TestCase
import unittest


def test_will_propagate(b, z, w):
    x1 = 3
    x = x1
    if b > 0:
        y = z + w
    else:
        y = 0
    a = 2 * x
    return a < b

def null_func(a,b,c,d):
    False

@numba.njit
def dummy_aliased_func(A):
    return A

def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)

def findLhsAssign(func_ir, var):
    for label, block in func_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if isinstance(inst, ir.Assign) and inst.target.name==var:
                return True

    return False

class TestRemoveDead(TestCase):

    _numba_parallel_test_ = False

    def test1(self):
        typingctx = cpu_target.typing_context
        targetctx = cpu_target.target_context
        test_ir = compiler.run_frontend(test_will_propagate)

        typingctx.refresh()
        targetctx.refresh()
        args = (types.int64, types.int64, types.int64)
        typemap, _, calltypes, _ = type_inference_stage(typingctx, targetctx, test_ir, args, None)
        remove_dels(test_ir.blocks)
        in_cps, out_cps = copy_propagate(test_ir.blocks, typemap)
        apply_copy_propagate(test_ir.blocks, in_cps, get_name_var_table(test_ir.blocks), typemap, calltypes)

        remove_dead(test_ir.blocks, test_ir.arg_names, test_ir)
        self.assertFalse(findLhsAssign(test_ir, "x"))

    def test2(self):
        def call_np_random_seed():
            np.random.seed(2)

        def seed_call_exists(func_ir):
            for inst in func_ir.blocks[0].body:
                if (isinstance(inst, ir.Assign) and
                    isinstance(inst.value, ir.Expr) and
                    inst.value.op == 'call' and
                    func_ir.get_definition(inst.value.func).attr == 'seed'):
                    return True
            return False

        test_ir = compiler.run_frontend(call_np_random_seed)
        remove_dead(test_ir.blocks, test_ir.arg_names, test_ir)
        self.assertTrue(seed_call_exists(test_ir))

    def run_array_index_test(self, func):
        A1 = np.arange(6).reshape(2,3)
        A2 = A1.copy()
        i = 0
        cfunc = njit(numba.typeof(A1), numba.typeof(i))(func)

        func(A1, i)
        cfunc(A2, i)
        np.testing.assert_array_equal(A1, A2)

    def test_alias_ravel(self):
        def func(A, i):
            B = A.ravel()
            B[i] = 3

        self.run_array_index_test(func)

    def test_alias_flat(self):
        def func(A, i):
            B = A.flat
            B[i] = 3

        self.run_array_index_test(func)

    def test_alias_transpose1(self):
        def func(A, i):
            B = A.T
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_transpose2(self):
        def func(A, i):
            B = A.transpose()
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_transpose3(self):
        def func(A, i):
            B = np.transpose(A)
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_reshape1(self):
        def func(A, i):
            B = np.reshape(A, (3,2))
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_reshape2(self):
        def func(A, i):
            B = A.reshape(3,2)
            B[i,0] = 3

        self.run_array_index_test(func)

    def test_alias_func_ext(self):
        def func(A, i):
            B = dummy_aliased_func(A)
            B[i, 0] = 3

        # save global state
        old_ext_handlers = alias_func_extensions.copy()
        try:
            alias_func_extensions[('dummy_aliased_func',
                'numba.tests.test_remove_dead')] = alias_ext_dummy_func
            self.run_array_index_test(func)
        finally:
            # recover global state
            ir_utils.alias_func_extensions = old_ext_handlers

    def test_rm_dead_rhs_vars(self):
        """make sure lhs variable of assignment is considered live if used in
        rhs (test for #6715).
        """
        def func():
            for i in range(3):
                a = (lambda j: j)(i)
                a = np.array(a)
            return a

        self.assertEqual(func(), numba.njit(func)())


class TestSSADeadBranchPrune(TestCase):
    """
    Test issues that required dead-branch-prune on SSA IR
    """
    def test_issue_9706(self):
        @njit
        def foo(x, y=None):
            if y is not None:
                return x + y
            else:
                y = x
                return x + y

        @njit
        def foo_manual_ssa(x, y=None):
            if y is not None:
                return x + y
            else:
                # avoid changing type of `y`
                y_ = x
                return x + y_

        self.assertEqual(foo(3, None), foo_manual_ssa(3, None))
        self.assertEqual(foo(3, 10), foo_manual_ssa(3, 10))

    def test_issue_6541(self):
        @njit
        def f(xs, out=None):
            N, = xs.shape
            if out is None:
                out = np.arange(N)
            else:
                assert np.all((0 <= out) & (out < N))
            out[:] = N
            return out

        expected = f(np.array([3, 1, 2]))
        out = np.arange(3, dtype='i8')
        got = f(np.array([3, 1, 2]), out=out)
        self.assertIs(got, out)
        self.assertPreciseEqual(got, expected)
        out = None
        got = f(np.array([3, 1, 2]), out=out)
        self.assertPreciseEqual(got, expected)

    def test_issue_7482(self):
        @njit
        def compute(smth, weights, default=0.0):
            if weights is None:
                return None

            if len(weights) == 0:
                return default

            idx = smth > weights
            weights = weights[idx]

            return default * weights

        self.assertIsNone(compute(smth=1, weights=None))
        kwargs = dict(smth=1, weights=np.arange(5), default=np.zeros(1))
        self.assertEqual(compute(**kwargs),
                         compute.py_func(**kwargs))

    def test_issue_5661(self):
        @njit
        def foo(a, b=None):
            if b is None:
                b = 1
            elif b < a:
                b += 1

            return a + b

        args_list = [
            (1, 2),
            (2, 1),
            (1,),
        ]
        for args in args_list:
            self.assertEqual(foo(*args), foo.py_func(*args))

        # Variation
        # https://github.com/numba/numba/issues/5661#issuecomment-697902475
        def make(decor=njit):
            @decor
            def inner(state):
                if state is None:
                    state = 0
                else:
                    state += 1
                return state

            @decor
            def fn():
                state = None
                for i in range(10):
                    state = inner(state)
                return state

            return fn()

        self.assertEqual(make(), make(lambda x: x))

    def test_issue_9742(self):
        CONST = 32

        @jit
        def foo():
            # This is a prune by value case, conditional is a compile time
            # evaluatable constant.
            conditional = CONST // 2
            collect = []
            while conditional:
                collect.append(conditional)
                conditional //= 2

            return collect

        self.assertEqual(foo(), foo.py_func())

    def test_issue_9742_variant(self):
        CONST = 32

        @jit
        def foo():
            collect = []
            # This is a prune by value case, conditional is a compile time
            # evaluatable constant.
            x = CONST + 1
            if x:
                collect.append(x)
            return collect

        self.assertEqual(foo(), foo.py_func())


if __name__ == "__main__":
    unittest.main()
