# coding: utf-8
"""Frontend collection for tree-lite"""
from __future__ import absolute_import
from .core import _LIB, c_str, _check_call
import ctypes

def _isascii(string):
  """Tests if a given string is pure ASCII; works for both Python 2 and 3"""
  try:
    return (len(string) == len(string.encode()))
  except UnicodeDecodeError:
    return False
  except UnicodeEncodeError:
    return False

class Model(object):
  """Decision tree ensemble model"""
  def __init__(self, handle=None):
    """
    Decision tree ensemble model

    Parameters
    ----------
    handle : `ctypes.c_void_p`, optional
        Initial value of model handle
    """
    if not isinstance(handle, ctypes.c_void_p):
      raise ValueError('Model handle must be of type ctypes.c_void_p')
    self.handle = handle

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreelitePredictorFree(self.handle))
      self.handle = None

def load_model_from_file(filename, format):
  if not _isascii(format):
    raise ValueError('format parameter must be an ASCII string')
  format = format.lower()
  if format == 'lightgbm':
    handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteLoadLightGBMModel(c_str(filename),
                                               ctypes.byref(handle)))
  elif format == 'xgboost':
    handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteLoadXGBoostModel(c_str(filename),
                                              ctypes.byref(handle)))
  elif format == 'protobuf':
    handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteLoadProtobufModel(c_str(filename),
                                               ctypes.byref(handle)))
  else:
    raise ValueError('Unknown format: must be one of ' \
                     + '{lightgbm, xgboost, protobuf}')
  model = Model(handle)
  return model
