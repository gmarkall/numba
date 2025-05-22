import re
import numpy as np

from numba.tests.support import TestCase, override_config, captured_stdout
from numba import jit, njit
from numba.core import types, ir, postproc, compiler
from numba.core.ir_utils import (guard, find_callname, find_const,
                                 get_definition, simplify_CFG)
from numba.core.registry import CPUDispatcher
from numba.core.inline_closurecall import inline_closure_call

from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
                             IRProcessing, DeadBranchPrune,
                             RewriteSemanticConstants, GenericRewrites,
                             WithLifting, PreserveIR, InlineClosureLikes)

from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
                                     NopythonRewrites, NativeLowering,
                                     IRLegalization, NoPythonBackend,
                                     NativeLowering)

from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
import unittest

@register_pass(analysis_only=False, mutates_CFG=True)
class InlineTestPass(FunctionPass):
    _name = "inline_test_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # assuming the function has one block with one call inside
        assert len(state.func_ir.blocks) == 1
        block = list(state.func_ir.blocks.values())[0]
        for i, stmt in enumerate(block.body):
            if guard(find_callname,state.func_ir, stmt.value) is not None:
                inline_closure_call(state.func_ir, {}, block, i, lambda: None,
                                    state.typingctx, state.targetctx, (),
                                    state.typemap, state.calltypes)
                break
        # also fix up the IR
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()
        post_proc.remove_dels()
        return True


def gen_pipeline(state, test_pass):
        name = 'inline_test'
        pm = PassManager(name)
        pm.add_pass(TranslateByteCode, "analyzing bytecode")
        pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")
        # pre typing
        if not state.flags.no_rewrites:
            pm.add_pass(GenericRewrites, "nopython rewrites")
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
        pm.add_pass(InlineClosureLikes,
                    "inline calls to locally defined closures")
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")

        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")

        pm.add_pass(test_pass, "inline test")

        # legalise
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(PreserveIR, "preserve IR")

        # lower
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(NoPythonBackend, "nopython mode backend")
        return pm

class InlineTestPipeline(compiler.CompilerBase):
    """compiler pipeline for testing inlining after optimization
    """
    def define_pipelines(self):
        pm = gen_pipeline(self.state, InlineTestPass)
        pm.finalize()
        return [pm]

class TestInlining(TestCase):
    """
    Check that jitted inner functions are inlined into outer functions,
    in nopython mode.
    Note that not all inner functions are guaranteed to be inlined.
    We just trust LLVM's inlining heuristics.
    """

    def make_pattern(self, fullname):
        """
        Make regexpr to match mangled name
        """
        parts = fullname.split('.')
        return r'_ZN?' + r''.join([r'\d+{}'.format(p) for p in parts])

    def assert_has_pattern(self, fullname, text):
        pat = self.make_pattern(fullname)
        self.assertIsNotNone(re.search(pat, text),
                             msg='expected {}'.format(pat))

    def assert_not_has_pattern(self, fullname, text):
        pat = self.make_pattern(fullname)
        self.assertIsNone(re.search(pat, text),
                          msg='unexpected {}'.format(pat))

    def test_inner_function(self):
        from numba.tests.inlining_usecases import outer_simple, \
            __name__ as prefix

        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_simple)
        self.assertPreciseEqual(cfunc(1), 4)
        # Check the inner function was elided from the output (which also
        # guarantees it was inlined into the outer function).
        asm = out.getvalue()
        self.assert_has_pattern('%s.outer_simple' % prefix, asm)
        self.assert_not_has_pattern('%s.inner' % prefix, asm)

    def test_multiple_inner_functions(self):
        from numba.tests.inlining_usecases import outer_multiple, \
            __name__ as prefix
        # Same with multiple inner functions, and multiple calls to
        # the same inner function (inner()).  This checks that linking in
        # the same library/module twice doesn't produce linker errors.
        with override_config('DUMP_ASSEMBLY', True):
            with captured_stdout() as out:
                cfunc = jit((types.int32,), nopython=True)(outer_multiple)
        self.assertPreciseEqual(cfunc(1), 6)
        asm = out.getvalue()
        self.assert_has_pattern('%s.outer_multiple' % prefix, asm)
        self.assert_not_has_pattern('%s.more' % prefix, asm)
        self.assert_not_has_pattern('%s.inner' % prefix, asm)


if __name__ == '__main__':
    unittest.main()
