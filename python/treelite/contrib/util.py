# coding: utf-8
"""
Miscellaneous utilities
"""

import shutil
import inspect
import time
from ..compat import PY3

if PY3:
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

__all__ = ['TemporaryDirectory', 'lineno']
