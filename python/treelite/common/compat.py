# coding: utf-8
# pylint: disable=W0611
"""Compatibility layer"""

from __future__ import absolute_import as _abs
import sys

PY3 = (sys.version_info[0] == 3)

def assert_python_min_ver(py2_ver, py3_ver, info_str):
  """Enforce minimum Python version for Python 2.x and 3.x"""
  py2_ver_ = py2_ver.split('.')
  py3_ver_ = py3_ver.split('.')
  if len(py2_ver_) != 2 or len(py3_ver_) != 2 or \
     py2_ver_[0] != '2' or py3_ver_[0] != '3':
    raise ValueError('Incorrect version format')
  if PY3:
    if sys.version_info[1] < int(py3_ver_[1]):
      raise RuntimeError('Python {} or newer is required. Feature: {}'\
                         .format(py3_ver, info_str))
  else:
    if sys.version_info[1] < int(py2_ver_[1]):
      raise RuntimeError('Python {} or newer is required. Feature: {}'\
                         .format(py2_ver, info_str))

# String handling for Python 2 and 3
if PY3:
  STRING_TYPES = (str,)
  def py_str(string):
    """Convert C string back to Python string"""
    return string.decode('utf-8')
  def _str_decode(string):
    return string.decode('utf-8')
  def _str_encode(string):
    return string.encode('utf-8')
else:
  STRING_TYPES = (basestring,)   # pylint: disable=E0602
  def py_str(string):
    """Convert C string back to Python string"""
    return string
  def _str_decode(string):
    return string
  def _str_encode(string):
    return string

# define DEVNULL
if PY3:
  from subprocess import DEVNULL
else:
  import os
  DEVNULL = open(os.devnull, 'r+b')

__all__ = []
