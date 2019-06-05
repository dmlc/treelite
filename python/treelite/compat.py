# coding: utf-8
# pylint: disable=W0611
"""Compatibility layer"""

from __future__ import absolute_import as _abs
import sys
import ctypes
from .common.compat import assert_python_min_ver, PY3

if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
  from json import JSONDecodeError   # Python 3.5 or newer
else:
  JSONDecodeError = ValueError

# expose C buffer as Python buffer
assert_python_min_ver('2.5', '3.3', 'buffer_from_memory')
if PY3:
  def buffer_from_memory(ptr, size):
    """Make Python buffer from raw memory"""
    func = ctypes.pythonapi.PyMemoryView_FromMemory
    func.restype = ctypes.py_object
    PyBUF_READ = 0x100         # pylint: disable=C0103
    return func(ptr, size, PyBUF_READ)
else:
  def buffer_from_memory(ptr, size):
    """Make Python buffer from raw memory"""
    func = ctypes.pythonapi.PyBuffer_FromMemory
    func.restype = ctypes.py_object
    return func(ptr, size)

# use cPickle if available
try:
  import cPickle as pickle
except ImportError:
  import pickle

# optional support for Pandas: if unavailable, define a dummy class
try:
  from pandas import DataFrame
  PANDAS_INSTALLED = True
except ImportError:
  class DataFrame():  # pylint: disable=R0903
    """dummy for pandas.DataFrame"""
  PANDAS_INSTALLED = False

__all__ = []
