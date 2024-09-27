"""
Miscellaneous utilities
"""

import ctypes
import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Any, Callable, TypeVar

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


_T = TypeVar("_T")


# Notice for `require_keyword_args`
# Authors: Olivier Grisel
#          Gael Varoquaux
#          Andreas Mueller
#          Lars Buitinck
#          Alexandre Gramfort
#          Nicolas Tresegnie
#          Sylvain Marie
# License: BSD 3 clause
def _require_keyword_args(
    error: bool,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning or error when passed as a positional argument.

    Modified from sklearn utils.validation.

    Parameters
    ----------
    error :
        Whether to throw an error or raise a warning.
    """

    def throw_if(func: Callable[..., _T]) -> Callable[..., _T]:
        """Throw an error/warning if there are positional arguments after the asterisk.

        Parameters
        ----------
        func :
            function to check arguments on.

        """
        sig = signature(func)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(func)
        def inner_f(*args: Any, **kwargs: Any) -> _T:
            extra_args = len(args) - len(all_args)
            if not all_args and extra_args > 0:  # keyword argument only
                raise TypeError("Keyword argument is required.")

            if extra_args > 0:
                # ignore first 'self' argument for instance methods
                args_msg = [
                    f"{name}"
                    for name, _ in zip(kwonly_args[:extra_args], args[-extra_args:])
                ]
                # pylint: disable=consider-using-f-string
                msg = "Pass `{}` as keyword args.".format(", ".join(args_msg))
                if error:
                    raise TypeError(msg)
                warnings.warn(msg, FutureWarning)
            for k, arg in zip(sig.parameters, args):
                kwargs[k] = arg
            return func(**kwargs)

        return inner_f

    return throw_if


deprecate_positional_args = _require_keyword_args(False)
