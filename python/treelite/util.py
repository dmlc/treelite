# coding: utf-8
"""
Miscellaneous utilities
"""
import inspect
import ctypes
import time
import numpy as np

_CTYPES_TYPE_TABLE = {
    'uint32': ctypes.c_uint32,
    'float32': ctypes.c_float,
    'float64': ctypes.c_double
}

_NUMPY_TYPE_TABLE = {
    'uint32': np.uint32,
    'float32': np.float32,
    'float64': np.float64
}

_NUMPY_TYPE_TABLE_INV = {
    np.uint32: 'unit32',
    np.float32: 'float32',
    np.float64: 'float64'
}


class TreeliteError(Exception):
    """Error thrown by Treelite"""


def buffer_from_memory(ptr, size):
    """Make Python buffer from raw memory"""
    func = ctypes.pythonapi.PyMemoryView_FromMemory
    func.restype = ctypes.py_object
    PyBUF_READ = 0x100  # pylint: disable=C0103
    return func(ptr, size, PyBUF_READ)


def c_str(string):
    """Convert a Python string to C string"""
    return ctypes.c_char_p(string.encode('utf-8'))


def py_str(string):
    """Convert C string back to Python string"""
    return string.decode('utf-8')


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python console"""
    print("{0:s}".format(py_str(msg)))


def lineno():
    """Returns line number"""
    return inspect.currentframe().f_back.f_lineno


def log_info(filename, linenum, msg):
    """Mimics behavior of the logging macro LOG(INFO) in dmlc-core"""
    print(f'[{time.strftime("%X")}] {filename}:{linenum}: {msg}')


def type_info_to_ctypes_type(type_info):
    """Obtain ctypes type corresponding to a given TypeInfo"""
    return _CTYPES_TYPE_TABLE[type_info]


def type_info_to_numpy_type(type_info):
    """Obtain ctypes type corresponding to a given TypeInfo"""
    return _NUMPY_TYPE_TABLE[type_info]


def numpy_type_to_type_info(type_info):
    """Obtain TypeInfo corresponding to a given NumPy type"""
    return _NUMPY_TYPE_TABLE_INV[type_info]
