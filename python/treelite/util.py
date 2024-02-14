"""
Miscellaneous utilities
"""

import ctypes

import numpy as np

_CTYPES_TYPE_TABLE = {
    "uint32": ctypes.c_uint32,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}


_NUMPY_TYPE_TABLE = {"uint32": np.uint32, "float32": np.float32, "float64": np.float64}


def typestr_to_ctypes_type(type_info):
    """Obtain ctypes type corresponding to a given Type str"""
    return _CTYPES_TYPE_TABLE[type_info]


def typestr_to_numpy_type(type_info):
    """Obtain ctypes type corresponding to a given Type str"""
    return _NUMPY_TYPE_TABLE[type_info]


def c_str(string):
    """Convert a Python string to C string"""
    return ctypes.c_char_p(string.encode("utf-8"))


def py_str(string):
    """Convert C string back to Python string"""
    return string.decode("utf-8")


def c_array(ctype, values):
    """
    Convert a Python byte array to C array

    WARNING
    -------
    DO NOT USE THIS FUNCTION if performance is critical. Instead, use np.array(*)
    with dtype option to explicitly convert type and then use
    ndarray.ctypes.data_as(*) to expose underlying buffer as C pointer.
    """
    return (ctype * len(values))(*values)
