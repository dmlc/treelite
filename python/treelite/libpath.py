# coding: utf-8
"""Find the path to Treelite dynamic library files."""

import os
import platform
import sys


class TreeliteLibraryNotFound(Exception):
    """Error thrown by when Treelite is not found"""


def find_lib_path():
    """Find the path to Treelite dynamic library files.

    Parameters
    ----------
    basename : :py:class:`str <python:str>`
        the base name of library
    libformat : boolean, optional (default True)
        if True, transform the base name to obtain the file name of the library
        ({}.dll on Windows; lib{}.so on Linux; lib{}.dylib on Mac OS X)
        if False, do not transform the base name at all; use it as a file name
        (this is useful to locate a file that's not a shared library)

    Returns
    -------
    lib_path: :py:class:`list <python:list>` of :py:class:`str <python:str>`
       List of all found library path to Treelite
    """
    if sys.platform == 'win32':
        lib_name = 'treelite.dll'
    elif sys.platform.startswith('linux'):
        lib_name = 'libtreelite.so'
    elif sys.platform == 'darwin':
        lib_name = 'libtreelite.dylib'
    else:
        raise RuntimeError('Unsupported operating system')

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # List possible locations for the library file
    dll_path = [
        # normal, after installation `lib` is copied into Python package tree.
        os.path.join(curr_path, 'lib'),
        # editable installation, no copying is performed.
        os.path.join(curr_path, os.path.pardir, os.path.pardir, 'build')
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
        candidate_list = '\n'.join([os.path.normpath(x) for x in dll_path])
        raise TreeliteLibraryNotFound(
            f'Cannot find library {lib_name} in the candidate path: ' +
            f'List of candidates:\n{candidate_list}\n')
    return lib_path


__all__ = []
