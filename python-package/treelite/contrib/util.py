# coding: utf-8
"""
Miscellaneous utilities
"""

from ..compat import PY3

import shutil
import inspect
import time

if PY3:
  from tempfile import TemporaryDirectory
else:
  class TemporaryDirectory(object):
    """Context manager for tempfile.mkdtemp()"""
    def __enter__(self):
        self.name = tempfile.mkdtemp()
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)

def lineno():
  """Returns line number"""
  return inspect.currentframe().f_back.f_lineno

def log_info(filename, linenum, msg):
  print('[{}] {}:{}: {}'.format(time.strftime('%X'), filename, linenum, msg))

__all__ = ['TemporaryDirectory', 'lineno']
