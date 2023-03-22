import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir

from numba.core import typing, types, debuginfo, itanium_mangler, cgutils
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.base import BaseContext
from numba.core.callconv import MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel

from .cudadrv import nvvm
from numba.cuda import codegen
from numba.cuda.models import cuda_data_manager

from warnings import warn

# -----------------------------------------------------------------------------
# Typing


class CUDATypingContext(typing.BaseContext):
    def load_additional_registries(self):
        from . import cudadecl, cudamath, libdevicedecl, vector_types
        from numba.core.typing import enumdecl, cffi_utils

        self.install_registry(cudadecl.registry)
        self.install_registry(cffi_utils.registry)
        self.install_registry(cudamath.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(libdevicedecl.registry)
        self.install_registry(enumdecl.registry)
        self.install_registry(vector_types.typing_registry)

    def resolve_value_type(self, val):
        # treat other dispatcher object as another device function
        from numba.cuda.dispatcher import CUDADispatcher
        if (isinstance(val, Dispatcher) and not
                isinstance(val, CUDADispatcher)):
            try:
                # use cached device function
                val = val.__dispatcher
            except AttributeError:
                if not val._can_compile:
                    raise ValueError('using cpu function on device '
                                     'but its compilation is disabled')
                targetoptions = val.targetoptions.copy()
                targetoptions['device'] = True
                targetoptions['debug'] = targetoptions.get('debug', False)
                targetoptions['opt'] = targetoptions.get('opt', True)
                disp = CUDADispatcher(val.py_func, targetoptions)
                # cache the device function for future use and to avoid
                # duplicated copy of the same function.
                val.__dispatcher = disp
                val = disp

        # continue with parent logic
        return super(CUDATypingContext, self).resolve_value_type(val)

# -----------------------------------------------------------------------------
# Implementation


VALID_CHARS = re.compile(r'[^a-z0-9]', re.I)


class CUDATargetContext(BaseContext):
    implement_powi_as_math_call = True
    strict_alignment = True

    def __init__(self, typingctx, target='cuda'):
        super().__init__(typingctx, target)
        self.data_model_manager = cuda_data_manager.chain(
            datamodel.default_manager
        )

    @property
    def DIBuilder(self):
        if nvvm.NVVM().is_nvvm70:
            return debuginfo.DIBuilder
        else:
            msg = "debuginfo is not generated for CUDA toolkits < 11.2"
            warn(NumbaInvalidConfigWarning(msg))
            return debuginfo.DummyDIBuilder

    @property
    def enable_boundscheck(self):
        # Unconditionally disabled
        return False

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    def init(self):
        self._internal_codegen = codegen.JITCUDACodegen("numba.cuda.jit")
        self._target_data = None

    def load_additional_registries(self):
        # side effect of import needed for numba.cpython.*, the builtins
        # registry is updated at import time.
        from numba.cpython import numbers, tupleobj, slicing # noqa: F401
        from numba.cpython import rangeobj, iterators, enumimpl # noqa: F401
        from numba.cpython import unicode, charseq # noqa: F401
        from numba.cpython import cmathimpl
        from numba.misc import cffiimpl
        from numba.np import arrayobj # noqa: F401
        from numba.np import npdatetime # noqa: F401
        from . import (
            cudaimpl, printimpl, libdeviceimpl, mathimpl, vector_types
        )

        self.install_registry(cudaimpl.registry)
        self.install_registry(cffiimpl.registry)
        self.install_registry(printimpl.registry)
        self.install_registry(libdeviceimpl.registry)
        self.install_registry(cmathimpl.registry)
        self.install_registry(mathimpl.registry)
        self.install_registry(vector_types.impl_registry)

    def codegen(self):
        return self._internal_codegen

    @property
    def target_data(self):
        if self._target_data is None:
            self._target_data = ll.create_target_data(nvvm.NVVM().data_layout)
        return self._target_data

    @cached_property
    def nonconst_module_attrs(self):
        """
        Some CUDA intrinsics are at the module level, but cannot be treated as
        constants, because they are loaded from a special register in the PTX.
        These include threadIdx, blockDim, etc.
        """
        from numba import cuda
        nonconsts = ('threadIdx', 'blockDim', 'blockIdx', 'gridDim', 'laneid',
                     'warpsize')
        nonconsts_with_mod = tuple([(types.Module(cuda), nc)
                                    for nc in nonconsts])
        return nonconsts_with_mod

    @cached_property
    def call_conv(self):
        return CUDACallConv(self)

    def mangler(self, name, argtypes, *, abi_tags=(), uid=None):
        return itanium_mangler.mangle(name, argtypes, abi_tags=abi_tags,
                                      uid=uid)

    def make_constant_array(self, builder, aryty, arr):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """

        lmod = builder.module

        constvals = [
            self.get_constant(types.byte, i)
            for i in iter(arr.tobytes(order='A'))
        ]
        constaryty = ir.ArrayType(ir.IntType(8), len(constvals))
        constary = ir.Constant(constaryty, constvals)

        addrspace = nvvm.ADDRSPACE_CONSTANT
        gv = cgutils.add_global_variable(lmod, constary.type, "_cudapy_cmem",
                                         addrspace=addrspace)
        gv.linkage = 'internal'
        gv.global_constant = True
        gv.initializer = constary

        # Preserve the underlying alignment
        lldtype = self.get_data_type(aryty.dtype)
        align = self.get_abi_sizeof(lldtype)
        gv.align = 2 ** (align - 1).bit_length()

        # Convert to generic address-space
        ptrty = ir.PointerType(ir.IntType(8))
        genptr = builder.addrspacecast(gv, ptrty, 'generic')

        # Create array object
        ary = self.make_array(aryty)(self, builder)
        kshape = [self.get_constant(types.intp, s) for s in arr.shape]
        kstrides = [self.get_constant(types.intp, s) for s in arr.strides]
        self.populate_array(ary, data=builder.bitcast(genptr, ary.data.type),
                            shape=kshape,
                            strides=kstrides,
                            itemsize=ary.itemsize, parent=ary.parent,
                            meminfo=None)

        return ary._getvalue()

    def insert_const_string(self, mod, string):
        """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """
        text = cgutils.make_bytearray(string.encode("utf-8") + b"\x00")
        name = '$'.join(["__conststring__",
                         itanium_mangler.mangle_identifier(string)])
        # Try to reuse existing global
        gv = mod.globals.get(name)
        if gv is None:
            # Not defined yet
            gv = cgutils.add_global_variable(mod, text.type, name,
                                             addrspace=nvvm.ADDRSPACE_CONSTANT)
            gv.linkage = 'internal'
            gv.global_constant = True
            gv.initializer = text

        # Cast to a i8* pointer
        charty = gv.type.pointee.element
        return gv.bitcast(charty.as_pointer(nvvm.ADDRSPACE_CONSTANT))

    def insert_string_const_addrspace(self, builder, string):
        """
        Insert a constant string in the constant addresspace and return a
        generic i8 pointer to the data.

        This function attempts to deduplicate.
        """
        lmod = builder.module
        gv = self.insert_const_string(lmod, string)
        charptrty = ir.PointerType(ir.IntType(8))
        return builder.addrspacecast(gv, charptrty, 'generic')

    def optimize_function(self, func):
        """Run O1 function passes
        """
        pass
        ## XXX skipped for now
        # fpm = lp.FunctionPassManager.new(func.module)
        #
        # lp.PassManagerBuilder.new().populate(fpm)
        #
        # fpm.initialize()
        # fpm.run(func)
        # fpm.finalize()


class CUDACallConv(MinimalCallConv):
    pass
