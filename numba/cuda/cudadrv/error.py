class CudaDriverError(Exception):
    pass


class NvvmError(Exception):
    def __str__(self):
        return '\n'.join(map(str, self.args))


class NvvmSupportError(ImportError):
    pass
