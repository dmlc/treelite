# coding: utf-8
"""Place holder for prediction runtime"""
import sys
import os
try:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../runtime/native/python/'))
  from treelite_runtime import *  # pylint: disable-msg=W0401
except ModuleNotFoundError:
  sys.path.insert(0, os.path.dirname(__file__))
  from treelite_runtime import *  # pylint: disable-msg=W0401
