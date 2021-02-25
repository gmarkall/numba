from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc

from numba.core import config
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm

import ctypes
import numpy as np


CUDA_TRIPLE = 'nvptx64-nvidia-cuda'


# Should the code library provide for multiple CCs? Or create one code library
# per CC?
class CUDACodeLibrary(CodeLibrary):

    def __init__(self, codegen, name):
        #from pudb import set_trace; set_trace()
        super().__init__(codegen, name)
        self._module = None
        self._linking_libraries = set()
        # XXX: Won't be able to serialize if we have linking files
        self._linking_files = set()

        # Caches
        # PTX cache keyed by CC: cc -> ptx string
        self._ptx_cache = {}
        # cubin cache: cc -> cubin
        self._cubin_cache = {}
        # compileinfo cache: cc -> compileinfo
        self._compileinfo_cache = {}
        # cufunc cache: device id -> cufunc
        self._cufunc_cache = {}

    def get_llvm_str(self):
        return str(self._module)

    # XXX: Need to make sure we can't fill the PTX cache here without getting a
    # cubin first, because its the cubin that has the nvvm options.
    # Alternatively, should configure a code library with nvvm options
    # immediately.
    def get_asm_str(self, cc=None, opt=None, options=None):
        #from pudb import set_trace; set_trace()
        if not cc:
            ctx = devices.get_context()
            device = ctx.device
            cc = device.compute_capability

        ptx = self._ptx_cache.get(cc, None)
        if ptx:
            print("PTX CACHE HIT", self.name)
            return ptx
        else:
            print("PTX CACHE MISS", self.name)

        arch = nvvm.get_arch_option(*cc)
        if opt is None:
            opt = 3
        if options is None:
            options = {}
        options['opt'] = opt
        options['arch'] = arch

        # XXX: Non-debug path - debug path requires separate compilation, TBC
        irs = [str(mod) for mod in self.modules]
        ptx = nvvm.llvm_to_ptx(irs, **options)
        ptx = ptx.decode().strip('\x00').strip()

        #from pudb import set_trace; set_trace()
        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % self._name).center(80, '-'))
            print(ptx)
            print('=' * 80)

        self._ptx_cache[cc] = ptx

        return ptx

    def get_cubin(self, max_registers=None, nvvm_options=None):
        if nvvm_options is None:
            nvvm_options = {}
        # XXX: Need caching for compute target
        #from pudb import set_trace; set_trace()
        ctx = devices.get_context()
        # XXX: Need caching of PTX
        device = ctx.device
        cc = device.compute_capability

        cubin = self._cubin_cache.get(cc, None)
        if cubin:
            print("CUBIN CACHE HIT")
            return cubin
        else:
            print("CUBIN CACHE MISS")

        ptx = self.get_asm_str(cc=cc, options=nvvm_options)
        linker = driver.Linker(max_registers=max_registers)
        linker.add_ptx(ptx.encode())
        #for lib in self._linking_libraries:
        #    linker.add_ptx(lib.get_asm_str().encode())
        for path in self._linking_files:
            linker.add_file_guess_ext(path)
        cubin_buf, size = linker.complete()
        compileinfo = linker.info_log
        # We take a copy of the cubin because it's owned by the linker
        cubin_ptr = ctypes.cast(cubin_buf, ctypes.POINTER(ctypes.c_char))
        cubin = bytes(np.ctypeslib.as_array(cubin_ptr, shape=(size,)))
        self._cubin_cache[cc] = cubin
        self._compileinfo_cache[cc] = compileinfo
        return cubin

    def get_cufunc(self, entry_name, max_registers=None, nvvm_options=None):
        ctx = devices.get_context()
        device = ctx.device

        cufunc = self._cufunc_cache.get(device.id, None)
        if cufunc:
            return cufunc

        cubin = self.get_cubin(max_registers=max_registers,
                               nvvm_options=nvvm_options)
        module = ctx.create_module_image(cubin)

        # Load
        cufunc = module.get_function(entry_name)

        # Populate caches
        self._cufunc_cache[device.id] = cufunc

        return cufunc

    def get_compileinfo(self, cc):
        try:
            return self._compileinfo_cache[cc]
        except KeyError:
            raise KeyError(f'No compileinfo for CC {cc}')

    def add_ir_module(self, mod):
        self._raise_if_finalized()
        if self._module is not None:
            raise RuntimeError('CUDACodeLibrary only supports one module')
        self._module = mod

    def add_linking_library(self, library):
        library._ensure_finalized()

        # We don't want to allow linking more libraries in after finalization
        # because our linked libraries are modified by the finalization, and we
        # won't be able to finalize again after adding new ones
        self._raise_if_finalized()

        self._linking_libraries.add(library)

    def add_linking_file(self, filepath):
        self._linking_files.add(filepath)

    def get_function(self, name):
        for fn in self._module.functions:
            if fn.name == name:
                return fn
        raise KeyError(f'Function {name} not found')

    @property
    def modules(self):
        return [self._module] + [mod for lib in self._linking_libraries
                                 for mod in lib.modules]

    def finalize(self):
        # Unlike the CPUCodeLibrary, we don't invoke the binding layer here -
        # we only adjust the linkage of functions. Global kernels (with
        # external linkage) have their linkage untouched. Device functions are
        # set linkonce_odr to prevent them appearing in the PTX.

        self._raise_if_finalized()

        # Note in-place modification of the linkage of functions in linked
        # libraries. This presently causes no issues as only device functions
        # are shared across code libraries, so they would always need their
        # linkage set to linkonce_odr. If in a future scenario some code
        # libraries require linkonce_odr linkage of functions in linked
        # modules, and another code library requires another linkage, each code
        # library will need to take its own private copy of its linked modules.
        #
        # See also discussion on PR #890:
        # https://github.com/numba/numba/pull/890
        for library in self._linking_libraries:
            for fn in library._module.functions:
                if not fn.is_declaration and fn.linkage != 'external':
                    fn.linkage = 'linkonce_odr'

        self._finalized = True


class JITCUDACodegen(Codegen):
    """
    This codegen implementation for CUDA only generates optimized LLVM IR.
    Generation of PTX code is done separately (see numba.cuda.compiler).
    """

    _library_class = CUDACodeLibrary

    def __init__(self, module_name):
        self._data_layout = nvvm.default_data_layout
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = lc.Module(name)
        ir_module.triple = CUDA_TRIPLE
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        nvvm.add_ir_version(ir_module)
        return ir_module

    def _add_module(self, module):
        pass
