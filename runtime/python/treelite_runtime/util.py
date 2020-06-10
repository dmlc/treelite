# coding: utf-8
"""
Miscellaneous utilities
"""
import inspect
import ctypes
import time
import sys
from sys import platform as _platform


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


def expand_windows_path(dirpath):
    """
    Expand a short path to full path (only applicable for Windows)
    Parameters
    ----------
    dirpath : :py:class:`str <python:str>`
        Path to expand
    Returns
    -------
    fullpath : :py:class:`str <python:str>`
        Expanded path
    """
    if sys.platform == 'win32':
        BUFFER_SIZE = 500
        buffer = ctypes.create_unicode_buffer(BUFFER_SIZE)
        get_long_path_name = ctypes.windll.kernel32.GetLongPathNameW
        get_long_path_name.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_long_path_name(dirpath, buffer, BUFFER_SIZE)
        return buffer.value
    return dirpath
