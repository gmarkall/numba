from __future__ import absolute_import, print_function, division

import contextlib
import logging
import sys
import weakref
from abc import ABCMeta, abstractmethod
from ctypes import byref, c_size_t, c_void_p
from collections import deque, namedtuple
from contextlib import contextmanager

from numba import config, utils, mviewbuf
from numba.utils import longint as long
from . import drvapi, enums
from .error import CudaAPIError


def _make_logger():
    logger = logging.getLogger(__name__)
    # is logging configured?
    if not utils.logger_hasHandlers(logger):
        # read user config
        lvl = str(config.CUDA_LOG_LEVEL).upper()
        lvl = getattr(logging, lvl, None)
        if not isinstance(lvl, int):
            # default to critical level
            lvl = logging.CRITICAL
        logger.setLevel(lvl)
        # did user specify a level?
        if config.CUDA_LOG_LEVEL:
            # create a simple handler that prints to stderr
            handler = logging.StreamHandler(sys.stderr)
            fmt = '== CUDA [%(relativeCreated)d] %(levelname)5s -- %(message)s'
            handler.setFormatter(logging.Formatter(fmt=fmt))
            logger.addHandler(handler)
        else:
            # otherwise, put a null handler
            logger.addHandler(logging.NullHandler())
    return logger


_logger = _make_logger()


class BaseCUDAMemoryManager(object, metaclass=ABCMeta):
    @abstractmethod
    def memalloc(self, nbytes, stream=0):
        pass

    @abstractmethod
    def memhostalloc(self, nbytes, mapped, portable, wc):
        pass

    @abstractmethod
    def mempin(self, owner, pointer, size, mapped):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_ipc_handle(self, ary, stream=0):
        pass

    @abstractmethod
    def get_memory_info(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def defer_cleanup(self):
        pass

    @property
    @abstractmethod
    def interface_version(self):
        pass


class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager):

    def __init__(self):
        self.allocations = utils.UniqueDict()
        self.deallocations = PendingDeallocs()

    def _attempt_allocation(self, allocator):
        """
        Attempt allocation by calling *allocator*.  If a out-of-memory error
        is raised, the pending deallocations are flushed and the allocation
        is retried.  If it fails in the second attempt, the error is reraised.
        """
        try:
            allocator()
        except CudaAPIError as e:
            # is out-of-memory?
            if e.code == enums.CUDA_ERROR_OUT_OF_MEMORY:
                # clear pending deallocations
                self.deallocations.clear()
                # try again
                allocator()
            else:
                raise

    def memhostalloc(self, bytesize, mapped=False, portable=False,
                     wc=False):
        pointer = c_void_p()
        flags = 0
        if mapped:
            flags |= enums.CU_MEMHOSTALLOC_DEVICEMAP
        if portable:
            flags |= enums.CU_MEMHOSTALLOC_PORTABLE
        if wc:
            flags |= enums.CU_MEMHOSTALLOC_WRITECOMBINED

        def allocator():
            driver_funcs.cuMemHostAlloc(byref(pointer), bytesize, flags)

        if mapped:
            self._attempt_allocation(allocator)
        else:
            allocator()

        owner = None

        finalizer = _hostalloc_finalizer(self, pointer, bytesize, mapped)

        if mapped:
            mem = MappedMemory(weakref.proxy(self), owner, pointer, bytesize,
                               finalizer=finalizer)
            self.allocations[mem.handle.value] = mem
            return mem.own()
        else:
            mem = PinnedMemory(weakref.proxy(self), owner, pointer, bytesize,
                               finalizer=finalizer)
            return mem

    def mempin(self, owner, pointer, size, mapped=False):
        if isinstance(pointer, (int, long)):
            pointer = c_void_p(pointer)

        # possible flags are "portable" (between context)
        # and "device-map" (map host memory to device thus no need
        # for memory transfer).
        flags = 0

        if mapped:
            flags |= enums.CU_MEMHOSTREGISTER_DEVICEMAP

        def allocator():
            driver_funcs.cuMemHostRegister(pointer, size, flags)

        if mapped:
            self._attempt_allocation(allocator)
        else:
            allocator()

        finalizer = _pin_finalizer(self, pointer, mapped)

        if mapped:
            mem = MappedMemory(weakref.proxy(self), owner, pointer, size,
                               finalizer=finalizer)
            self.allocations[mem.handle.value] = mem
            return mem.own()
        else:
            mem = PinnedMemory(weakref.proxy(self), owner, pointer, size,
                               finalizer=finalizer)
            return mem

    def initialize(self):
        # XXX: Need to have a pending deallocs list that doesn't need the size
        # of device memory to be constructed.
        pass

    def reset(self):
        self.allocations.clear()
        self.deallocations.clear()

    @contextmanager
    def defer_cleanup(self):
        with self.deallocations.disable():
            yield


_MemoryInfo = namedtuple("_MemoryInfo", "free,total")


class NumbaCUDAMemoryManager(HostOnlyCUDAMemoryManager):
    def initialize(self):
        # setup *deallocations* as the memory manager becomes active for the
        # first time
        if self.deallocations.memory_capacity == _SizeNotSet:
            self.deallocations.memory_capacity = self.get_memory_info().total

    def memalloc(self, bytesize):
        ptr = drvapi.cu_device_ptr()

        def allocator():
            driver_funcs.cuMemAlloc(byref(ptr), bytesize)

        self._attempt_allocation(allocator)

        finalizer = _alloc_finalizer(self, ptr, bytesize)
        # XXX: Should the context be used here or the memory manager?
        #      - probably the context and not the memory manager, since its
        #      just used for the memory pointer to store the context it was
        #      allocated in.
        mem = AutoFreePointer(weakref.proxy(self), ptr, bytesize, finalizer)
        self.allocations[ptr.value] = mem
        return mem.own()

    def get_memory_info(self):
        free = c_size_t()
        total = c_size_t()
        driver_funcs.cuMemGetInfo(byref(free), byref(total))
        return _MemoryInfo(free=free.value, total=total.value)

    def get_ipc_handle(self, memory, stream=0):
        ipchandle = drvapi.cu_ipc_mem_handle()
        driver_funcs.cuIpcGetMemHandle(
            byref(ipchandle),
            memory.owner.handle,
        )
        from numba import cuda
        source_info = cuda.current_context().device.get_device_identity()
        offset = memory.handle.value - memory.owner.handle.value

        from numba.cuda.cudadrv.driver import IpcHandle
        return IpcHandle(memory, ipchandle, memory.size, source_info,
                         offset=offset)

    @property
    def interface_version(self):
        return 1


class DriverFuncs(object):
    def __init__(self):
        self._ready = False

    def _ensure_ready(self):
        from .driver import driver
        self._driver = driver
        self._ready = True

    def __getattr__(self, fname):
        self._ensure_ready()
        return self._driver.__getattr__(fname)


driver_funcs = DriverFuncs()


def _alloc_finalizer(memory_manager, handle, size):
    allocations = memory_manager.allocations
    deallocations = memory_manager.deallocations

    def core():
        if allocations:
            del allocations[handle.value]
        deallocations.add_item(driver_funcs.cuMemFree, handle, size)

    return core


def _hostalloc_finalizer(memory_manager, handle, size, mapped):
    """
    Finalize page-locked host memory allocated by `context.memhostalloc`.

    This memory is managed by CUDA, and finalization entails deallocation. The
    issues noted in `_pin_finalizer` are not relevant in this case, and the
    finalization is placed in the `context.deallocations` queue along with
    finalization of device objects.

    """
    allocations = memory_manager.allocations
    deallocations = memory_manager.deallocations
    if not mapped:
        size = _SizeNotSet

    def core():
        if mapped and allocations:
            del allocations[handle.value]
        deallocations.add_item(driver_funcs.cuMemFreeHost, handle, size)

    return core


def _pin_finalizer(memory_manager, handle, mapped):
    """
    Finalize temporary page-locking of host memory by `context.mempin`.

    This applies to memory not otherwise managed by CUDA. Page-locking can
    be requested multiple times on the same memory, and must therefore be
    lifted as soon as finalization is requested, otherwise subsequent calls to
    `mempin` may fail with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, leading
    to unexpected behavior for the context managers `cuda.{pinned,mapped}`.
    This function therefore carries out finalization immediately, bypassing the
    `context.deallocations` queue.

    """
    allocations = memory_manager.allocations

    def core():
        if mapped and allocations:
            del allocations[handle.value]
        driver_funcs.cuMemHostUnregister(handle)

    return core


class MemoryPointer(object):

    """A memory pointer that owns the buffer with an optional finalizer.

    When an instance is deleted, the finalizer will be called regardless
    of the `.refct`.

    An instance is created with `.refct=1`.  The buffer lifetime
    is tied to the MemoryPointer instance's lifetime.  The finalizer is invoked
    only if the MemoryPointer instance's lifetime ends.
    """
    __cuda_memory__ = True

    def __init__(self, context, pointer, size, finalizer=None, owner=None):
        self.context = context
        self.device_pointer = pointer
        self.size = size
        self._cuda_memsize_ = size
        self.is_managed = finalizer is not None
        self.refct = 1
        self.handle = self.device_pointer
        self._owner = owner

        if finalizer is not None:
            self._finalizer = utils.finalize(self, finalizer)

    @property
    def owner(self):
        return self if self._owner is None else self._owner

    def own(self):
        return OwnedPointer(weakref.proxy(self))

    def free(self):
        """
        Forces the device memory to the trash.
        """
        if self.is_managed:
            if not self._finalizer.alive:
                raise RuntimeError("Freeing dead memory")
            self._finalizer()
            assert not self._finalizer.alive

    def memset(self, byte, count=None, stream=0):
        count = self.size if count is None else count
        if stream:
            driver_funcs.cuMemsetD8Async(self.device_pointer, byte, count,
                                         stream.handle)
        else:
            driver_funcs.cuMemsetD8(self.device_pointer, byte, count)

    def view(self, start, stop=None):
        if stop is None:
            size = self.size - start
        else:
            size = stop - start

        # Handle NULL/empty memory buffer
        if self.device_pointer.value is None:
            if size != 0:
                raise RuntimeError("non-empty slice into empty slice")
            view = self      # new view is just a reference to self
        # Handle normal case
        else:
            base = self.device_pointer.value + start
            if size < 0:
                raise RuntimeError('size cannot be negative')
            pointer = drvapi.cu_device_ptr(base)
            view = MemoryPointer(self.context, pointer, size, owner=self.owner)

        if isinstance(self.owner, (MemoryPointer, OwnedPointer)):
            # Owned by a numba-managed memory segment, take an owned reference
            return OwnedPointer(weakref.proxy(self.owner), view)
        else:
            # Owned by external alloc, return view with same external owner
            return view

    @property
    def device_ctypes_pointer(self):
        return self.device_pointer


class OwnedPointer(object):
    def __init__(self, memptr, view=None):
        self._mem = memptr

        if view is None:
            self._view = self._mem
        else:
            assert not view.is_managed
            self._view = view

        mem = self._mem

        def deref():
            try:
                mem.refct -= 1
                assert mem.refct >= 0
                if mem.refct == 0:
                    mem.free()
            except ReferenceError:
                # ignore reference error here
                pass

        self._mem.refct += 1
        utils.finalize(self, deref)

    def __getattr__(self, fname):
        """Proxy MemoryPointer methods
        """
        return getattr(self._view, fname)


class AutoFreePointer(MemoryPointer):
    """Modifies the ownership semantic of the MemoryPointer so that the
    instance lifetime is directly tied to the number of references.

    When `.refct` reaches zero, the finalizer is invoked.
    """
    def __init__(self, *args, **kwargs):
        super(AutoFreePointer, self).__init__(*args, **kwargs)
        # Releease the self reference to the buffer, so that the finalizer
        # is invoked if all the derived pointers are gone.
        self.refct -= 1


class MappedMemory(AutoFreePointer):
    __cuda_memory__ = True

    def __init__(self, context, owner, hostpointer, size,
                 finalizer=None):
        self.owned = owner
        self.host_pointer = hostpointer
        devptr = drvapi.cu_device_ptr()
        driver_funcs.cuMemHostGetDevicePointer(byref(devptr), hostpointer, 0)
        self.device_pointer = devptr
        super(MappedMemory, self).__init__(context, devptr, size,
                                           finalizer=finalizer)
        self.handle = self.host_pointer

        # For buffer interface
        self._buflen_ = self.size
        self._bufptr_ = self.host_pointer.value

    def own(self):
        return MappedOwnedPointer(weakref.proxy(self))


class PinnedMemory(mviewbuf.MemAlloc):
    def __init__(self, context, owner, pointer, size, finalizer=None):
        self.context = context
        self.owned = owner
        self.size = size
        self.host_pointer = pointer
        self.is_managed = finalizer is not None
        self.handle = self.host_pointer

        # For buffer interface
        self._buflen_ = self.size
        self._bufptr_ = self.host_pointer.value

        if finalizer is not None:
            utils.finalize(self, finalizer)

    def own(self):
        return self


class MappedOwnedPointer(OwnedPointer, mviewbuf.MemAlloc):
    pass


class _SizeNotSet(int):
    """
    Dummy object for PendingDeallocs when *size* is not set.
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, 0)

    def __str__(self):
        return '?'


_SizeNotSet = _SizeNotSet()


class PendingDeallocs(object):
    """
    Pending deallocations of a context (or device since we are using the primary
    context).
    """
    def __init__(self, capacity=_SizeNotSet):
        self._cons = deque()
        self._disable_count = 0
        self._size = 0
        self.memory_capacity = capacity

    @property
    def _max_pending_bytes(self):
        return int(self.memory_capacity * config.CUDA_DEALLOCS_RATIO)

    def add_item(self, dtor, handle, size=_SizeNotSet):
        """
        Add a pending deallocation.

        The *dtor* arg is the destructor function that takes an argument,
        *handle*.  It is used as ``dtor(handle)``.  The *size* arg is the
        byte size of the resource added.  It is an optional argument.  Some
        resources (e.g. CUModule) has an unknown memory footprint on the device.
        """
        _logger.info('add pending dealloc: %s %s bytes', dtor.__name__, size)
        self._cons.append((dtor, handle, size))
        self._size += int(size)
        if (len(self._cons) > config.CUDA_DEALLOCS_COUNT or
                self._size > self._max_pending_bytes):
            self.clear()

    def clear(self):
        """
        Flush any pending deallocations unless it is disabled.
        Do nothing if disabled.
        """
        if not self.is_disabled:
            while self._cons:
                [dtor, handle, size] = self._cons.popleft()
                _logger.info('dealloc: %s %s bytes', dtor.__name__, size)
                dtor(handle)
            self._size = 0

    @contextlib.contextmanager
    def disable(self):
        """
        Context manager to temporarily disable flushing pending deallocation.
        This can be nested.
        """
        self._disable_count += 1
        try:
            yield
        finally:
            self._disable_count -= 1
            assert self._disable_count >= 0

    @property
    def is_disabled(self):
        return self._disable_count > 0

    def __len__(self):
        """
        Returns number of pending deallocations.
        """
        return len(self._cons)
