import ctypes
import functools
import logging
import os
import sys

from itertools import product
from numba.core import config, utils
from numba.cuda.cudart.rtapi import API_PROTOTYPES
from numba.cuda.cudart import enums
from numba.cuda.envvars import get_numba_envvar
from numba.cuda.errors import CudaSupportError, CudaRuntimeError


def _make_logger():
    logger = logging.getLogger(__name__)
    # is logging configured?
    if not logger.hasHandlers():
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


def find_runtime():

    envpath = get_numba_envvar('CUDA_RUNTIME')

    if envpath == '0':
        # Force fail
        _raise_runtime_not_found()

    # Determine DLL type
    if sys.platform == 'win32':
        dlloader = ctypes.WinDLL
        dldir = ['\\windows\\system32']
        dlnames = ['cudart.dll']
    else:
        # Assume to be *nix like
        dlloader = ctypes.CDLL
        dldir = ['/usr/lib', '/usr/local/cuda/lib64'] # FIXME - path
        dlnames = ['libcudart.so']

    if envpath is not None:
        try:
            envpath = os.path.abspath(envpath)
        except ValueError:
            raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid path" %
                             envpath)
        if not os.path.isfile(envpath):
            raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid file "
                             "path.  Note it must be a filepath of the .so/"
                             ".dll/.dylib or the driver" % envpath)
        candidates = [envpath]
    else:
        # First search for the name in the default library path.
        # If that is not found, try the specific path.
        candidates = dlnames + [os.path.join(x, y)
                                for x, y in product(dldir, dlnames)]

    # Load the driver; Collect driver error information
    path_not_exist = []
    driver_load_error = []

    for path in candidates:
        try:
            dll = dlloader(path)
        except OSError as e:
            # Problem opening the DLL
            path_not_exist.append(not os.path.isfile(path))
            driver_load_error.append(e)
        else:
            return dll

    # Problem loading driver
    if all(path_not_exist):
        _raise_runtime_not_found()
    else:
        errmsg = '\n'.join(str(e) for e in driver_load_error)
        _raise_driver_error(errmsg)


RUNTIME_NOT_FOUND_MSG = """
CUDA runtime library cannot be found.
If you are sure that a CUDA runtime is installed,
try setting environment variable NUMBA_CUDA_RUNTIME
with the file path of the CUDA runtime shared library.
"""

RUNTIME_LOAD_ERROR_MSG = """
Possible CUDA rtunime libraries are found but error occurred during load:
%s
"""

MISSING_FUNCTION_ERRMSG = "runtime missing function: %s."


def _build_reverse_error_map():
    prefix = 'cuda'
    map = utils.UniqueDict()
    for name in dir(enums):
        if name.startswith(prefix):
            code = getattr(enums, name)
            map[code] = name
    return map


def _getpid():
    return os.getpid()


ERROR_MAP = _build_reverse_error_map()


def _raise_runtime_not_found():
    raise CudaSupportError(RUNTIME_NOT_FOUND_MSG)


def _raise_driver_error(e):
    raise CudaSupportError(RUNTIME_LOAD_ERROR_MSG % e)


class CudaRuntimeAPIError(CudaRuntimeError):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super().__init__(code, msg)

    def __str__(self):
        return "[%s] %s" % (self.code, self.msg)


class Runtime:
    """
    Runtime object that lazily binds runtime API functions.
    """

    def __init__(self):
        self.is_initialized = False
        try:
            if config.DISABLE_CUDA:
                msg = ("CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1 "
                       "in the environment, or because CUDA is unsupported on "
                       "32-bit systems.")
                raise CudaSupportError(msg)
            self.lib = find_runtime()
            self.load_error = None
        except CudaSupportError as e:
            self.load_error = e

    def initialize(self):
        # lazily initialize logger
        global _logger
        _logger = _make_logger()

    def __getattr__(self, fname):
        # First request of a runtime API function
        try:
            proto = API_PROTOTYPES[fname]
        except KeyError:
            raise AttributeError(fname)
        restype = proto[0]
        argtypes = proto[1:]

        if not self.is_initialized:
            self.initialize()

        if self.load_error is not None:
            raise CudaSupportError("Error at runtime load: \n%s:" %
                                   self.load_error)

        # Find function in runtime library
        libfn = self._find_api(fname)
        libfn.restype = restype
        libfn.argtypes = argtypes

        safe_call = self._wrap_api_call(fname, libfn)
        setattr(self, fname, safe_call)
        return safe_call

    def _wrap_api_call(self, fname, libfn):
        @functools.wraps(libfn)
        def safe_cuda_api_call(*args):
            _logger.debug('call runtime api: %s', libfn.__name__)
            retcode = libfn(*args)
            self._check_error(fname, retcode)
        return safe_cuda_api_call

    def _check_error(self, fname, retcode):
        if retcode != enums.cudaSuccess:
            errname = ERROR_MAP.get(retcode, "cudaErrorUnknown")
            msg = "Call to %s results in %s" % (fname, errname)
            _logger.error(msg)
            raise CudaRuntimeAPIError(retcode, msg)

    def _find_api(self, fname):
        try:
            return getattr(self.lib, fname)
        except AttributeError:
            pass

        # Not found.
        # Delay missing function error to use
        def absent_function(*args, **kws):
            raise CudaRuntimeError(MISSING_FUNCTION_ERRMSG % fname)

        setattr(self, fname, absent_function)
        return absent_function

    def get_version(self):
        """
        Returns the CUDA Runtime version as a tuple (major, minor).
        """
        rtver = ctypes.c_int()
        self.cudaRuntimeGetVersion(ctypes.byref(rtver))
        # The version is encoded as (1000 * major) + (10 * minor)
        major = rtver.value // 1000
        minor = (rtver.value - (major * 1000)) // 10
        return (major, minor)


runtime = Runtime()
