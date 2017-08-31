# coding: utf-8
"""compiler module"""
from __future__ import absolute_import as _abs
from .core import _LIB, c_str, _check_call
from .frontend import Model
import ctypes
import collections

class Compiler(object):
  """
  Compiler object to translate a tree ensemble model into a semantic model
  """
  
  def __init__(self, name='recursive'):
    """Compiler object to translate a tree ensemble model into a semantic model

    Parameters
    ----------
    name : string, optional
        name of compiler (default: 'recursive')
    """
    self.handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteCompilerCreate(c_str(name),
                                            ctypes.byref(self.handle)))

  def compile(self, model, dirpath, params, verbose=False):
    """
    Generate prediction code from a tree ensemble model. The code will be C99
    compliant. One header file (.h) will be generated, along with one or more
    source files (.c). Use contrib.*.create_shared() to package prediction code
    as a dynamic shared library (.so/.dll/.dylib).

    Usage
    -----
    compiler.compile(model, dirpath="./my/model", verbose=True);
    # files to generate: ./my/model/model.h, ./my/model/model.c
    # if parallel compilation is enabled:
    # ./my/model/model.h, ./my/model/model0.c, ./my/model/model1.c,
    # ./my/model/model2.c, and so forth

    Parameters
    ----------
    model : `Model`
        decision tree ensemble model
    dirpath : string
        directory to store header and source files
    params : dict
        parameters for compiler
    verbose : boolean, optional
        Whether to print extra messages during compilation
    """
    _params = dict(params) if isinstance(params, list) else params
    self._set_param(_params or {})
    if not isinstance(model, Model):
      raise ValueError('model parameter must be of Model type')
    _check_call(_LIB.TreeliteCompilerGenerateCode(self.handle,
                                                  model.handle,
                                                  ctypes.c_int(verbose),
                                                  c_str(dirpath)))

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreeliteCompilerFree(self.handle))
      self.handle = None

  def _set_param(self, params, value=None):
    """
    Set parameter(s)

    Parameters
    ----------
    params: dict / list / string
        list of key-alue pairs, dict or simply string key
    value: optional
        value of the specified parameter, when params is a single string
    """
    if isinstance(params, collections.Mapping):
      params = params.items()
    elif isinstance(params, STRING_TYPES) and value is not None:
      params = [(params, value)]
    for key, val in params:
      _check_call(_LIB.TreeliteCompilerSetParam(self.handle, c_str(key),
                                                c_str(str(val))))

__all__ = ['Compiler']
