"""Optional FP32 matrix multiplication using the ROCm HIP and rocBLAS C APIs."""

import ctypes as ct
import os
from pathlib import Path

import numpy as np


PTR = ct.c_void_p
INT = ct.c_int
SIZE = ct.c_size_t
FLOAT_PTR = ct.POINTER(ct.c_float)


def bind(library, name, *types):
    function = getattr(library, name)
    function.argtypes = types
    function.restype = INT

    def call(*args):
        status = function(*args)
        if status:
            raise RuntimeError(f"{name} failed with status {status}")

    return call


class Rocblas:
    """Reuse device buffers while keeping training and tensor ownership in NumPy."""

    def __init__(self):
        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "lib"
        hip = ct.CDLL(str(root / "libamdhip64.so"))
        blas = ct.CDLL(str(root / "librocblas.so"))
        self.allocate = bind(hip, "hipMalloc", ct.POINTER(PTR), SIZE)
        self.free = bind(hip, "hipFree", PTR)
        self.copy = bind(hip, "hipMemcpy", PTR, PTR, SIZE, INT)
        self.destroy = bind(blas, "rocblas_destroy_handle", PTR)
        self.gemm = bind(blas, "rocblas_sgemm", PTR, INT, INT, INT, INT, INT,
                         FLOAT_PTR, PTR, INT, PTR, INT, FLOAT_PTR, PTR, INT)
        self.handle = PTR()
        bind(blas, "rocblas_create_handle", ct.POINTER(PTR))(ct.byref(self.handle))
        self.buffers = [(PTR(), 0) for _ in range(3)]
        self.one = ct.c_float(1)
        self.zero = ct.c_float(0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            for pointer, size in self.buffers:
                if size:
                    self.free(pointer)
        finally:
            self.destroy(self.handle)

    def buffer(self, index, size):
        pointer, capacity = self.buffers[index]
        if capacity < size:
            replacement = PTR()
            self.allocate(ct.byref(replacement), size)
            if capacity:
                self.free(pointer)
            pointer = replacement
            self.buffers[index] = (pointer, size)
        return pointer

    def __call__(self, left, right):
        if left.dtype != np.float32 or right.dtype != np.float32:
            raise ValueError("The ROCm backend supports FP32 only.")
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        rows, inner = left.shape
        right_rows, columns = right.shape
        if inner != right_rows or min(rows, inner, columns) == 0:
            raise ValueError("Expected two compatible, nonempty matrices.")
        output = np.empty((rows, columns), dtype=np.float32)
        a, b, c = (
            self.buffer(index, array.nbytes)
            for index, array in enumerate((left, right, output))
        )
        self.copy(a, left.ctypes.data, left.nbytes, 1)
        self.copy(b, right.ctypes.data, right.nbytes, 1)
        # rocBLAS is column-major. The same buffers hold the transposed matrices,
        # so compute output.T = right.T @ left.T.
        no_transpose = 111
        self.gemm(self.handle, no_transpose, no_transpose, columns, rows, inner,
                  ct.byref(self.one), b, columns, a, inner,
                  ct.byref(self.zero), c, columns)
        self.copy(output.ctypes.data, c, output.nbytes, 2)
        return output
