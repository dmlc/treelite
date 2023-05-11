# coding: utf-8
"""
Miscellaneous utilities
"""
import ctypes
import warnings

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


class TreeliteError(Exception):
    """Error thrown by Treelite"""


def c_str(string):
    """Convert a Python string to C string"""
    return ctypes.c_char_p(string.encode('utf-8'))


def py_str(string):
    """Convert C string back to Python string"""
    return string.decode('utf-8')


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python console"""
    print(py_str(msg))


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _warn_callback(msg: bytes) -> None:
    """Redirect warnings from native library into Python console"""
    warnings.warn(py_str(msg))


def type_info_to_ctypes_type(type_info):
    """Obtain ctypes type corresponding to a given TypeInfo"""
    return _CTYPES_TYPE_TABLE[type_info]


def type_info_to_numpy_type(type_info):
    """Obtain ctypes type corresponding to a given TypeInfo"""
    return _NUMPY_TYPE_TABLE[type_info]
