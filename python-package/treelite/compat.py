# coding: utf-8
"""Compatibility layer"""

from __future__ import absolute_import as _abs

import sys
import ctypes

PY3 = (sys.version_info[0] == 3)

# String handling for Python 2 and 3
if PY3:
  STRING_TYPES = str,
  def py_str(x):
    """Convert C string back to Python string"""
    return x.decode('utf-8')
  def _str_decode(str):
    return str.decode('utf-8')
  def _str_encode(str):
    return str.encode('utf-8')
else:
  STRING_TYPES = basestring,
  def py_str(x):
    """Convert C string back to Python string"""
    return x
  def _str_decode(str):
    return str
  def _str_encode(str):
    return str

if PY3:
  from json import JSONDecodeError
else:
  JSONDecodeError = ValueError

# expose C buffer as Python buffer
if PY3:
  if sys.version_info[1] < 3:
    raise RuntimeError('Python 3.3 or newer is required.')
  def buffer_from_memory(ptr, size):
    func = ctypes.pythonapi.PyMemoryView_FromMemory
    func.restype = ctypes.py_object
    PyBUF_READ = 0x100
    return func(ptr, size, PyBUF_READ)
else:
  def buffer_from_memory(ptr, size):
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
  class DataFrame(object):
    """dummy for pandas.DataFrame"""
    pass
  PANDAS_INSTALLED = False

__all__ = ['']
