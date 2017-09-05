# coding: utf-8
"""predictor module"""

from .core import c_str, _get_log_callback_func, TreeliteError, DMatrix
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

class Batch(object):
  """batch of rows to be used for prediction"""
  def __init__(self):
    self.handle = None
    self.kind = None

  def __del__(self):
    if self.handle is not None:
      if self.kind == 'sparse':
        _check_call(_LIB.TreeliteDeleteSparseBatch(self.handle))
      elif self.kind == 'dense':
        _check_call(_LIB.TreeliteDeleteDenseBatch(self.handle))
      else:
        raise TreeliteError('this batch has wrong value for `kind` field')
      self.handle = None
      self.kind = None

  def shape(self):
    """
    Get dimensions of the batch

    Returns
    -------
    tuple (number of rows, number of columns)
    """
    num_row = ctypes.c_size_t()
    num_col = ctypes.c_size_t()
    _check_call(_LIB.TreeliteBatchGetDimension(self.handle,
                               ctypes.c_int(1 if self.kind == 'sparse' else 0),
                                               ctypes.byref(num_row),
                                               ctypes.byref(num_col)))
    return (num_row.value, num_col.value)

  @classmethod
  def from_npy2d(self, mat, rbegin, rend=None, missing=None):
    """
    Get a dense batch from a 2D numpy matrix.
    If ``mat`` does not have ``order='C'`` (also known as row-major) or is not
    contiguous, a temporary copy will be made.
    If ``mat`` does not have ``dtype=numpy.float32``, a temporary copy will be
    made also.
    Thus, as many as two temporary copies of data can be made. One should set
    input layout and type judiciously to conserve memory.

    Parameters
    ----------
    mat : 2D numpy array
        data matrix
    rbegin : integer
        the index of the first row in the subset
    rend : integer, optional (defaults to the end of matrix)
        one past the index of the last row in the subset
    missing : float, optional (defaults to np.nan)
        value indicating missing value
    Returns
    -------
    a dense batch consisting of rows [rbegin, rend)
    """
    if not isinstance(mat, np.ndarray):
      raise ValueError('mat must be of type numpy.ndarray')
    if len(mat.shape) != 2:
      raise ValueError('Input numpy.ndarray must be two-dimensional')
    rend = rend if rend is not None else mat.shape[0]
    if rbegin >= rend:
      raise TreeliteError('rbegin must be less than rend')
    if rbegin < 0:
      raise TreeliteError('rbegin must be nonnegative')
    if rend > mat.shape[0]:
      raise TreeliteError('rend must be less than number of rows in mat')
    # flatten the array by rows and ensure it is float32.
    # we try to avoid data copies if possible
    # (reshape returns a view when possible and we explicitly tell np.array to
    #  avoid copying)
    data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
    missing = missing if missing is not None else np.nan

    batch = Batch()
    batch.handle = ctypes.c_void_p()
    batch.kind = 'dense'
    _check_call(_LIB.TreeliteAssembleDenseBatch(
            data[rbegin:].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(missing),
            ctypes.c_size_t(rend - rbegin),
            ctypes.c_size_t(mat.shape[1]),
            ctypes.byref(batch.handle)))
    return batch

  @classmethod
  def from_dmatrix(cls, dmat, rbegin, rend=None):
    """
    Get a sparse batch from a subset of matrix rows. The subset is given by
    the range [rbegin, rend).
    
    Parameters
    ----------
    dmat : object of class `DMatrix`
        data matrix
    rbegin : integer
        the index of the first row in the subset
    rend : integer, optional (defaults to the end of matrix)
        one past the index of the last row in the subset

    Returns
    -------
    a sparse batch consisting of rows [rbegin, rend)
    """
    if not isinstance(dmat, DMatrix):
      raise ValueError('dmat must be of type DMatrix')
    # get internal members of data matrix
    data, col_ind, row_ptr, num_row, num_col, nelem = dmat._get_internals()
    rend = rend if rend is not None else num_row
    if rbegin >= rend:
      raise TreeliteError('rbegin must be less than rend')
    if rbegin < 0:
      raise TreeliteError('rbegin must be nonnegative')
    if rend > num_row:
      raise TreeliteError('rend must be less than number of rows in dmat')
    
    buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
    buffer_from_memory.restype = ctypes.py_object

    data_subset = np.frombuffer(buffer_from_memory(data,
                                     ctypes.sizeof(ctypes.c_float) \
                                       * (row_ptr[rend] - row_ptr[rbegin])),
                                np.float32)
    col_ind_subset = np.frombuffer(buffer_from_memory(col_ind,
                                        ctypes.sizeof(ctypes.c_uint32) \
                                          * (row_ptr[rend] - row_ptr[rbegin])),
                                   np.uint32)
    # need to re-compute row_ptr for subset; store new copy to Python object
    buffer = buffer_from_memory(row_ptr,
                                ctypes.sizeof(ctypes.c_size_t) * (num_row + 1))
    self.row_ptr_subset \
      = np.copy(np.frombuffer(buffer, np.uintp)[rbegin:(rend+1)])
    self.row_ptr_subset -= self.row_ptr_subset[0]

    batch = Batch()
    batch.handle = ctypes.c_void_p()
    batch.kind = 'sparse'
    _check_call(_LIB.TreeliteAssembleSparseBatch(
                                     c_array(ctypes.c_float, data_subset),
                                     c_array(ctypes.c_uint32, col_ind_subset),
                                     c_array(ctypes.c_size_t, row_ptr_subset),
                                     ctypes.c_size_t(num_row),
                                     ctypes.c_size_t(num_col),
                                     ctypes.byref(batch.handle)))
    return batch

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

  def predict(self, batch, nthread=None, verbose=False, pred_margin=False):
    """
    Make prediction using a batch of data rows

    Parameters
    ----------
    batch: object of type `Batch`
        batch of rows for which predictions will be made
    nthread : integer, optional
        Number of threads (default to number of cores)
    verbose : boolean, optional (default to False)
        Whether to print extra messages during prediction
    pred_margin: boolean, optional (default to False)
        whether to produce raw margins rather than transformed probabilities
    """
    if not isinstance(batch, Batch):
      raise TreeliteError('batch must be of type Batch')
    if batch.handle is None or batch.kind is None:
      raise TreeliteError('batch cannot be empty')
    nthread = nthread if nthread is not None else 0
    result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorQueryResultSize(self.handle,
                                                      batch.handle,
                              ctypes.c_int(1 if batch.kind == 'sparse' else 0),
                                                    ctypes.byref(result_size)))
    out_result = np.zeros(result_size.value, dtype=np.float32, order='C')
    out_result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorPredictBatch(self.handle, batch.handle,
                    ctypes.c_int(1 if batch.kind == 'sparse' else 0),
                    ctypes.c_int(nthread), ctypes.c_int(1 if verbose else 0),
                    ctypes.c_int(1 if pred_margin else 0),
                    out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.byref(out_result_size)))
    idx = out_result_size.value
    return out_result[0:idx].reshape((batch.shape()[0], -1)).squeeze()

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreelitePredictorFree(self.handle))
      self.handle = None

__all__ = ['Predictor', 'Batch']
