from __future__ import print_function, absolute_import, division


class CudaDriverError(Exception):
    pass


class CudaSupportError(ImportError):
    pass


class CudaAPIError(CudaDriverError):
    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        super(CudaAPIError, self).__init__(code, msg)

    def __str__(self):
        return "[%s] %s" % (self.code, self.msg)


class NvvmError(Exception):
    def __str__(self):
        return '\n'.join(map(str, self.args))


class NvvmSupportError(ImportError):
    pass
