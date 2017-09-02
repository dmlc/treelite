# coding: utf-8
"""predictor module"""

from .core import c_str, _get_log_callback_func, DMatrix, TreeliteError
from .libpath import find_lib_path
from .contrib.util import lineno, log_info
import ctypes
import os
import numpy as np

def _load_runtime_lib():
  """Load tree-lite runtime"""
  lib_path = find_lib_path(runtime=True)
  if len(lib_path) == 0:
    return None
  lib = ctypes.cdll.LoadLibrary(lib_path[0])
  lib.TreeliteGetLastError.restype = ctypes.c_char_p
  lib.callback = _get_log_callback_func()
  if lib.TreeliteRegisterLogCallback(lib.callback) != 0:
    raise TreeliteError(lib.TreeliteGetLastError())
  return lib

# load the tree-lite runtime
_LIB = _load_runtime_lib()

def _check_call(ret):
  """Check the return value of C API call

  This function will raise exception when error occurs.
  Wrap every API call with this function

  Parameters
  ----------
  ret : int
      return value from API calls
  """
  if ret != 0:
    raise TreeliteError(_LIB.TreeliteGetLastError())

class Predictor(object):
  """Predictor class"""
  def __init__(self, libpath, verbose=False):
    """
    Predictor class: load a compiled shared library

    Parameters
    ----------
    libpath: string
        location of dynamic shared library (.dll/.so/.dylib)
    verbose : boolean, optional (default to False)
        Whether to print extra messages during construction
    """
    if os.path.isdir(libpath):  # libpath is a diectory
      # directory is given; locate shared library inside it
      basename = os.path.basename(libpath.rstrip('/\\'))
      lib_found = False
      for ext in ['.so', '.dll', '.dylib']:
        path = os.path.join(libpath, basename + ext)
        if os.path.exists(path):
          lib_found = True
          break
      if not lib_found:
        raise TreeliteError('Directory {} doesn\'t appear '.format(libpath)+\
                            'to have any dynamic shared library '+\
                            '(.so/.dll/.dylib).')
    else:      # libpath is actually the name of shared library file
      fileext = os.path.splitext(libpath)[1]
      if fileext == '.dll' or fileext == '.so' or fileext == '.dylib':
        path = libpath
      else:
        raise TreeliteError('Specified path {} has wrong '.format(libpath) + \
                            'file extension ({}); '.format(fileext) +\
                            'the share library must have one of the '+\
                            'following extensions: .so / .dll / .dylib')
    self.handle = ctypes.c_void_p()
    _check_call(_LIB.TreelitePredictorLoad(c_str(path),
                                           ctypes.byref(self.handle)))
    if verbose:
      log_info(__file__, lineno(),
               'Dynamic shared library {} has been '.format(path)+\
               'successfully loaded into memory')

  def predict(self, dmat, nthread=None, verbose=False, pred_margin=False):
    """
    Make prediction using a data matrix

    Parameters
    ----------
    dmat: object of type `DMatrix`
        data matrix for which predictions will be made
    nthread : integer, optional
        Number of threads (default to number of cores)
    verbose : boolean, optional (default to False)
        Whether to print extra messages during prediction
    pred_margin: boolean, optional (default to False)
        whether to produce raw margins rather than transformed probabilities
    """
    if not isinstance(dmat, DMatrix):
      raise TreeliteError('dmat must be of type DMatrix')
    nthread = nthread if nthread is not None else 0
    result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorQueryResultSize(self.handle,
                                                      dmat.handle,
                                                    ctypes.byref(result_size)))
    out_result = np.zeros(result_size.value, dtype=np.float32, order='C')
    if pred_margin:
      _check_call(_LIB.TreelitePredictorPredictRaw(self.handle, dmat.handle,
                                                   ctypes.c_int(nthread),
                                             ctypes.c_int(1 if verbose else 0),
                    out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
      return out_result.reshape((dmat.shape()[0], -1)).squeeze()
    else:
      out_result_size = ctypes.c_size_t()
      _check_call(_LIB.TreelitePredictorPredict(self.handle, dmat.handle,
                                                ctypes.c_int(nthread),
                                             ctypes.c_int(1 if verbose else 0),
                     out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                                ctypes.byref(out_result_size)))
      idx = out_result_size.value
      return out_result[0:idx].reshape((dmat.shape()[0], -1)).squeeze()

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreelitePredictorFree(self.handle))
      self.handle = None

__all__ = ['Predictor']
