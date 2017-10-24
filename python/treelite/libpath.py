# coding: utf-8
"""Find the path to treelite dynamic library files."""

import os
import platform
import sys

class TreeliteLibraryNotFound(Exception):
  """Error thrown by when treelite is not found"""
  pass


def find_lib_path(runtime=False):
  """Find the path to treelite dynamic library files.

  Parameters
  ----------
  runtime : boolean, optional (default False)
      whether to load the runtime instead of the main library

  Returns
  -------
  lib_path: list(string)
     List of all found library path to treelite
  """
  lib_name = 'treelite_runtime' if runtime else 'treelite'

  curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
  # make pythonpack hack: copy this directory one level upper for setup.py
  dll_path = [curr_path, os.path.join(curr_path, '../../lib/'),
              os.path.join(curr_path, './lib/'),
              os.path.join(sys.prefix, 'treelite')]
  if sys.platform == 'win32':
    if platform.architecture()[0] == '64bit':
      dll_path.append(os.path.join(curr_path, '../../windows/x64/Release/'))
      # hack for pip installation when copy all parent source directory here
      dll_path.append(os.path.join(curr_path, './windows/x64/Release/'))
    else:
      dll_path.append(os.path.join(curr_path, '../../windows/Release/'))
      # hack for pip installation when copy all parent source directory here
      dll_path.append(os.path.join(curr_path, './windows/Release/'))
    dll_path = [os.path.join(p, '{}.dll'.format(lib_name)) for p in dll_path]
  elif sys.platform.startswith('linux'):
    dll_path = [os.path.join(p, 'lib{}.so'.format(lib_name)) for p in dll_path]
  elif sys.platform == 'darwin':
    dll_path = [os.path.join(p, 'lib{}.dylib'.format(lib_name)) \
                for p in dll_path]

  lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]

  if not lib_path:
    raise TreeliteLibraryNotFound(
        'Cannot find treelite library in the candidate path: ' +
        'List of candidates:\n' + ('\n'.join(dll_path)))
  return lib_path

__all__ = []
