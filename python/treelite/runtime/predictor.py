# coding: utf-8
# pylint: disable=W0201
"""predictor module"""
import ctypes
import sys
import os
import numpy as np
from ..common.util import c_str, _get_log_callback_func, TreeliteError, \
                          lineno, log_info
from ..common.libpath import find_lib_path, TreeliteLibraryNotFound

def _load_runtime_lib():
  """Load treelite runtime"""
  lib_path = find_lib_path(basename='treelite_runtime')
  lib = ctypes.cdll.LoadLibrary(lib_path[0])
  lib.TreeliteGetLastError.restype = ctypes.c_char_p
  lib.callback = _get_log_callback_func()
  if lib.TreeliteRegisterLogCallback(lib.callback) != 0:
    raise TreeliteError(lib.TreeliteGetLastError())
  return lib

# load the treelite runtime
# (do not load if called by sphinx)
if 'sphinx' in sys.modules:
  try:
    _LIB = _load_runtime_lib()
  except TreeliteLibraryNotFound:
    _LIB = None
else:
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
  """Batch of rows to be used for prediction"""
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
    dims : :py:class:`tuple <python:tuple>` of length 2
        (number of rows, number of columns)
    """
    num_row = ctypes.c_size_t()
    num_col = ctypes.c_size_t()
    _check_call(_LIB.TreeliteBatchGetDimension(
        self.handle,
        ctypes.c_int(1 if self.kind == 'sparse' else 0),
        ctypes.byref(num_row),
        ctypes.byref(num_col)))
    return (num_row.value, num_col.value)

  @classmethod
  def from_npy2d(cls, mat, rbegin=0, rend=None, missing=None):
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
    mat : object of type :py:class:`numpy.ndarray`, with dimension 2
        data matrix
    rbegin : :py:class:`int <python:int>`, optional
        the index of the first row in the subset
    rend : :py:class:`int <python:int>`, optional
        one past the index of the last row in the subset. If missing, set to
        the end of the matrix.
    missing : :py:class:`float <python:float>`, optional
        value indicating missing value. If missing, set to ``numpy.nan``.

    Returns
    -------
    dense_batch : :py:class:`Batch`
        a dense batch consisting of rows ``[rbegin, rend)``
    """
    if not isinstance(mat, np.ndarray):
      raise ValueError('mat must be of type numpy.ndarray')
    if len(mat.shape) != 2:
      raise ValueError('Input numpy.ndarray must be two-dimensional')
    num_row = mat.shape[0]
    num_col = mat.shape[1]
    rbegin = rbegin if rbegin is not None else 0
    rend = rend if rend is not None else num_row
    if rbegin >= rend:
      raise TreeliteError('rbegin must be less than rend')
    if rbegin < 0:
      raise TreeliteError('rbegin must be nonnegative')
    if rend > num_row:
      raise TreeliteError('rend must be less than number of rows in mat')
    # flatten the array by rows and ensure it is float32.
    # we try to avoid data copies if possible
    # (reshape returns a view when possible and we explicitly tell np.array to
    #  avoid copying)
    data_subset = np.array(mat[rbegin:rend, :].reshape((rend - rbegin)*num_col),
                           copy=False, dtype=np.float32)
    missing = missing if missing is not None else np.nan

    batch = Batch()
    batch.handle = ctypes.c_void_p()
    batch.kind = 'dense'
    _check_call(_LIB.TreeliteAssembleDenseBatch(
        data_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(missing),
        ctypes.c_size_t(rend - rbegin),
        ctypes.c_size_t(num_col),
        ctypes.byref(batch.handle)))
    # save handles for internal arrays
    batch.data = data_subset
    return batch

  @classmethod
  def from_csr(cls, csr, rbegin=None, rend=None):
    """
    Get a sparse batch from a subset of rows in a CSR (Compressed Sparse Row)
    matrix. The subset is given by the range ``[rbegin, rend)``.

    Parameters
    ----------
    csr : object of class :py:class:`treelite.DMatrix` or \
          :py:class:`scipy.sparse.csr_matrix`
        data matrix
    rbegin : :py:class:`int <python:int>`, optional
        the index of the first row in the subset
    rend : :py:class:`int <python:int>`, optional
        one past the index of the last row in the subset. If missing, set to
        the end of the matrix.

    Returns
    -------
    sparse_batch : :py:class:`Batch`
        a sparse batch consisting of rows ``[rbegin, rend)``
    """
    # use duck typing so as to accomodate both scipy.sparse.csr_matrix
    # and DMatrix without explictly importing any of them
    try:
      num_row = csr.shape[0]
      num_col = csr.shape[1]
    except AttributeError:
      raise ValueError('csr must contain shape attribute')
    except TypeError:
      raise ValueError('csr.shape must be of tuple type')
    except IndexError:
      raise ValueError('csr.shape must be of length 2 (indicating 2D matrix)')
    rbegin = rbegin if rbegin is not None else 0
    rend = rend if rend is not None else num_row
    if rbegin >= rend:
      raise TreeliteError('rbegin must be less than rend')
    if rbegin < 0:
      raise TreeliteError('rbegin must be nonnegative')
    if rend > num_row:
      raise TreeliteError('rend must be less than number of rows in csr')

    # compute submatrix with rows [rbegin, rend)
    ibegin = csr.indptr[rbegin]
    iend = csr.indptr[rend]
    data_subset = np.array(csr.data[ibegin:iend], copy=False,
                           dtype=np.float32, order='C')
    indices_subset = np.array(csr.indices[ibegin:iend], copy=False,
                              dtype=np.uint32, order='C')
    indptr_subset = np.array(csr.indptr[rbegin:(rend+1)] - ibegin, copy=False,
                             dtype=np.uintp, order='C')

    batch = Batch()
    batch.handle = ctypes.c_void_p()
    batch.kind = 'sparse'
    _check_call(_LIB.TreeliteAssembleSparseBatch(
        data_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        indptr_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(rend - rbegin),
        ctypes.c_size_t(num_col),
        ctypes.byref(batch.handle)))
    # save handles for internal arrays
    batch.data = data_subset
    batch.indices = indices_subset
    batch.indptr = indptr_subset
    return batch

class Predictor(object):
  """
  Predictor class: loader for compiled shared libraries

  Parameters
  ----------
  libpath: :py:class:`str <python:str>`
      location of dynamic shared library (.dll/.so/.dylib)
  nthread: :py:class:`int <python:int>`, optional
      number of worker threads to use; if unspecified, use maximum number of
      hardware threads
  verbose : :py:class:`bool <python:bool>`, optional
      Whether to print extra messages during construction
  include_master_thread : :py:class:`bool <python:bool>`, optional
      Whether to assign work to the master thread
  """
  # pylint: disable=R0903

  def __init__(self, libpath, nthread=None, verbose=False,
               include_master_thread=True):
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
    path = os.path.abspath(path)
    _check_call(_LIB.TreelitePredictorLoad(
        c_str(path),
        ctypes.c_int(nthread if nthread is not None else -1),
        ctypes.c_int(1 if include_master_thread else 0),
        ctypes.byref(self.handle)))
    if verbose:
      log_info(__file__, lineno(),
               'Dynamic shared library {} has been '.format(path)+\
               'successfully loaded into memory')

  def predict(self, batch, verbose=False, pred_margin=False):
    """
    Make prediction using a batch of data rows (synchronously). This will
    internally split workload among worker threads.

    Parameters
    ----------
    batch: object of type :py:class:`Batch`
        batch of rows for which predictions will be made
    verbose : :py:class:`bool <python:bool>`, optional
        Whether to print extra messages during prediction
    pred_margin: :py:class:`bool <python:bool>`, optional
        whether to produce raw margins rather than transformed probabilities
    """
    if not isinstance(batch, Batch):
      raise TreeliteError('batch must be of type Batch')
    if batch.handle is None or batch.kind is None:
      raise TreeliteError('batch cannot be empty')
    result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorQueryResultSize(
        self.handle,
        batch.handle,
        ctypes.c_int(1 if batch.kind == 'sparse' else 0),
        ctypes.byref(result_size)))
    out_result = np.zeros(result_size.value, dtype=np.float32, order='C')
    out_result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorPredictBatch(
        self.handle,
        batch.handle,
        ctypes.c_int(1 if batch.kind == 'sparse' else 0),
        ctypes.c_int(1 if verbose else 0),
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
