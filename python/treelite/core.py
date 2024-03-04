"""Interface with native lib"""

import ctypes
import os
import sys
import warnings

from .libpath import TreeliteLibraryNotFound, find_lib_path
from .util import py_str


class TreeliteError(Exception):
    """Error thrown by Treelite"""


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _log_callback(msg: bytes) -> None:
    """Redirect logs from native library into Python console"""
    print(py_str(msg))


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _warn_callback(msg: bytes) -> None:
    """Redirect warnings from native library into Python console"""
    warnings.warn(py_str(msg))


def _load_lib():
    """Load Treelite Library."""
    lib_path = [str(x) for x in find_lib_path()]
    if not lib_path:
        # Building docs
        return None  # type: ignore
    if sys.version_info >= (3, 8) and sys.platform == "win32":
        # pylint: disable=no-member
        os.add_dll_directory(
            os.path.join(os.path.normpath(sys.base_prefix), "Library", "bin")
        )
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.TreeliteGetLastError.restype = ctypes.c_char_p
    lib.log_callback = _log_callback
    lib.warn_callback = _warn_callback
    if lib.TreeliteRegisterLogCallback(lib.log_callback) != 0:
        raise TreeliteError(py_str(lib.TreeliteGetLastError()))
    if lib.TreeliteRegisterWarningCallback(lib.warn_callback) != 0:
        raise TreeliteError(py_str(lib.TreeliteGetLastError()))
    return lib


# Load the Treelite library globally
# (do not load if called by Sphinx)
if "sphinx" in sys.modules:
    try:
        _LIB = _load_lib()
    except TreeliteLibraryNotFound:
        _LIB = None
else:
    _LIB = _load_lib()


def _check_call(ret: int) -> None:
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function.

    Parameters
    ----------
    ret :
        return value from API calls
    """
    if ret != 0:
        raise TreeliteError(_LIB.TreeliteGetLastError().decode("utf-8"))
