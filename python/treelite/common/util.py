# coding: utf-8
"""
Miscellaneous utilities
"""
from __future__ import absolute_import as _abs
import ctypes
import inspect
import time
import os
import sys
import site
from .compat import py_str

class TreeliteVersionNotFound(Exception):
  """Error thrown by when version file is not found"""

def c_str(string):
  """Convert a Python string to C string"""
  return ctypes.c_char_p(string.encode('utf-8'))

def _log_callback(msg):
  """Redirect logs from native library into Python console"""
  print("{0:s}".format(py_str(msg)))

def _get_log_callback_func():
  """Wrap log_callback() method in ctypes callback type"""
  #pylint: disable=invalid-name
  CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
  return CALLBACK(_log_callback)

def _load_ver():
  curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
  # go one level up, as this script is in common/ directory
  curr_path = os.path.abspath(os.path.join(curr_path, os.pardir))
  # List possible locations for VERSION
  ver_path = [curr_path, os.path.join(curr_path, '../../'),
              os.path.join(sys.prefix, 'treelite'),
              os.path.join(site.USER_BASE, 'treelite')]
  ver_path = [os.path.join(p, 'VERSION') for p in ver_path]
  ver_path_found = [p for p in ver_path if os.path.exists(p) and os.path.isfile(p)]
  if not ver_path_found:
    raise TreeliteVersionNotFound(
        'Cannot find version information in the candidate path: ' +
        'List of candidates:\n' + ('\n'.join(ver_path)))
  with open(ver_path_found[0], 'r') as f:
    VERSION = f.readlines()[0].rstrip('\n')
  return VERSION

class TreeliteError(Exception):
  """Error thrown by treelite"""

def lineno():
  """Returns line number"""
  return inspect.currentframe().f_back.f_lineno

def log_info(filename, linenum, msg):
  """Mimics behavior of the logging macro LOG(INFO) in dmlc-core"""
  print('[{}] {}:{}: {}'.format(time.strftime('%X'), filename, linenum, msg))

__all__ = []
