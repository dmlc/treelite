# coding: utf-8
"""
Miscellaneous utilities
"""
import inspect
import ctypes
import time
from sys import platform as _platform
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


def type_info_to_ctypes_type(type_info):
    """Obtain ctypes type corresponding to a given TypeInfo"""
    return _CTYPES_TYPE_TABLE[type_info]


def type_info_to_numpy_type(type_info):
    """Obtain ctypes type corresponding to a given TypeInfo"""
    return _NUMPY_TYPE_TABLE[type_info]


def numpy_type_to_type_info(type_info):
    """Obtain TypeInfo corresponding to a given NumPy type"""
    if type_info == np.uint32:
        return 'uint32'
    elif type_info == np.float32:
        return 'float32'
    elif type_info == np.float64:
        return 'float64'
    else:
        raise ValueError(f'Unrecognized NumPy type: {type_info}')


class TreeliteRuntimeError(Exception):
    """Error thrown by Treelite runtime"""


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


def lib_extension_current_platform():
    if _platform == 'darwin':
        return '.dylib'
    if _platform in ('win32', 'cygwin'):
        return '.dll'
    return '.so'
