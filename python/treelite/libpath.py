# coding: utf-8
"""Find the path to treelite dynamic library files."""

import os
import platform
import sys
import site

class TreeliteLibraryNotFound(Exception):
  """Error thrown by when treelite is not found"""


def find_lib_path(basename, libformat=True):
  """Find the path to treelite dynamic library files.

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
     List of all found library path to treelite
  """
  if libformat:
    if sys.platform == 'win32':
      lib_name = '{}.dll'.format(basename)
    elif sys.platform.startswith('linux'):
      lib_name = 'lib{}.so'.format(basename)
    elif sys.platform == 'darwin':
      lib_name = 'lib{}.dylib'.format(basename)
    else:
      raise RuntimeError('Unsupported operating system')
  else:
    lib_name = basename

  curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
  # List possible locations for the library file
  dll_path = [curr_path,
              os.path.join(curr_path, '../../lib/'),
              os.path.join(curr_path, '../../build/lib/'),
              os.path.join(curr_path, '../../runtime/native/lib/'),
              os.path.join(curr_path, './lib/'),
              os.path.join(curr_path, './build/lib/'),
              os.path.join(sys.prefix, 'treelite'),
              os.path.join(site.USER_BASE, 'treelite')]
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
    raise TreeliteLibraryNotFound(
        'Cannot find library {} in the candidate path: '.format(lib_name) +
        'List of candidates:\n' + ('\n'.join(dll_path)))
  return lib_path

__all__ = []
