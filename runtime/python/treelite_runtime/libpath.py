# coding: utf-8
"""Find the path to Treelite dynamic library files."""

import os
import platform
import sys


class TreeliteRuntimeLibraryNotFound(Exception):
    """Error thrown by when Treelite runtime is not found"""


def find_lib_path():
    """Find the path to Treelite runtime library files.

    Returns
    -------
    lib_path: :py:class:`list <python:list>` of :py:class:`str <python:str>`
       List of all found library path to Treelite
    """
    if sys.platform == 'win32':
        lib_name = f'treelite_runtime.dll'
    elif sys.platform.startswith('linux'):
        lib_name = f'libtreelite_runtime.so'
    elif sys.platform == 'darwin':
        lib_name = f'libtreelite_runtime.dylib'
    else:
        raise RuntimeError('Unsupported operating system')

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # List possible locations for the library file
    dll_path = [
        # normal, after installation `lib` is copied into Python package tree.
        os.path.join(curr_path, 'lib'),
        # editable installation, no copying is performed.
        os.path.join(curr_path, os.path.pardir, os.path.pardir, os.path.pardir, 'build')
    ]
    # Windows hack: additional candidate locations
    if sys.platform == 'win32':
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
        else:
            dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
            # hack for pip installation when copy all parent source directory here
            dll_path.append(os.path.join(curr_path, './windows/Release/'))
    # Now examine all candidate locations for the library file
    dll_path = [os.path.join(p, lib_name) for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

    if not lib_path:
        candidate_list = '\n'.join(dll_path)
        raise TreeliteRuntimeLibraryNotFound(
            f'Cannot find library {lib_name} in the candidate path: ' +
            f'List of candidates:\n{os.path.normpath(candidate_list)}\n')
    return lib_path


__all__ = []
