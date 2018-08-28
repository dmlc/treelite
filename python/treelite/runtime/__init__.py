# coding: utf-8
"""Treelite prediction runtime package"""
import sys
import os
try:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../runtime/native/python/'))
  from treelite_runtime import *  # pylint: disable-msg=W0401
  import treelite_runtime as _t
  __all__ = _t.__all__
except ImportError:
  sys.path.insert(0, os.path.dirname(__file__))
  from treelite_runtime import *  # pylint: disable-msg=W0401
  import treelite_runtime as _t
  __all__ = _t.__all__
