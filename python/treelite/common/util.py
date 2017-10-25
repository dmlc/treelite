# coding: utf-8
"""
Miscellaneous utilities
"""
from __future__ import absolute_import as _abs
import ctypes
import inspect
import time
import shutil
from .compat import py_str, PY3

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

class TreeliteError(Exception):
  """Error thrown by treelite"""
  pass

if PY3:
  # pylint: disable=W0611
  from tempfile import TemporaryDirectory
else:
  import tempfile
  class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp()"""
    # pylint: disable=R0903

    def __enter__(self):
      self.name = tempfile.mkdtemp()    # pylint: disable=W0201
      return self.name

    def __exit__(self, exc_type, exc_value, traceback):
      shutil.rmtree(self.name)

def lineno():
  """Returns line number"""
  return inspect.currentframe().f_back.f_lineno

def log_info(filename, linenum, msg):
  """Mimics behavior of the logging macro LOG(INFO) in dmlc-core"""
  print('[{}] {}:{}: {}'.format(time.strftime('%X'), filename, linenum, msg))

__all__ = []
