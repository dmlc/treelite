# coding: utf-8
# pylint: disable=W0201
"""predictor module"""
import ctypes
import sys
import os
import re
import numpy as np
import scipy.sparse
from .common.util import c_str, _get_log_callback_func, TreeliteError, \
                          lineno, log_info, _load_ver
from .libpath import TreeliteLibraryNotFound, find_lib_path

__version__ = _load_ver()

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

class PredictorEntry(ctypes.Union):
  _fields_ = [('missing', ctypes.c_int), ('fvalue', ctypes.c_float)]

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
    # save pointer to mat so that it doesn't get garbage-collected prematurely
    batch.mat = mat
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
    # save pointer to csr so that it doesn't get garbage-collected prematurely
    batch.csr = csr
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
  """
  # pylint: disable=R0903

  def __init__(self, libpath, nthread=None, verbose=False):
    if os.path.isdir(libpath):  # libpath is a directory
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
    if not re.match(r'^[a-zA-Z]+://', path):
      path = os.path.abspath(path)
    _check_call(_LIB.TreelitePredictorLoad(
        c_str(path),
        ctypes.c_int(nthread if nthread is not None else -1),
        ctypes.byref(self.handle)))
    # save # of features
    num_feature = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorQueryNumFeature(
        self.handle,
        ctypes.byref(num_feature)))
    self.num_feature_ = num_feature.value
    # save # of output groups
    num_output_group = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorQueryNumOutputGroup(
        self.handle,
        ctypes.byref(num_output_group)))
    self.num_output_group_ = num_output_group.value
    # save # of pred transform
    pred_transform = ctypes.c_char_p()
    _check_call(_LIB.TreelitePredictorQueryPredTransform(
        self.handle,
        ctypes.byref(pred_transform)))
    self.pred_transform_ = bytes.decode(pred_transform.value)
    # save # of sigmoid alpha
    sigmoid_alpha = ctypes.c_float()
    _check_call(_LIB.TreelitePredictorQuerySigmoidAlpha(
        self.handle,
        ctypes.byref(sigmoid_alpha)))
    self.sigmoid_alpha_ = sigmoid_alpha.value
    # save # of global bias
    global_bias = ctypes.c_float()
    _check_call(_LIB.TreelitePredictorQueryGlobalBias(
        self.handle,
        ctypes.byref(global_bias)))
    self.global_bias_ = global_bias.value

    if verbose:
      log_info(__file__, lineno(),
               'Dynamic shared library {} has been '.format(path)+\
               'successfully loaded into memory')

  def predict_instance(self, inst, missing=None, pred_margin=False):
    """
    Perform single-instance prediction. Prediction is run by the calling thread.

    Parameters
    ----------
    inst: :py:class:`numpy.ndarray` / :py:class:`scipy.sparse.csr_matrix` /\
          :py:class:`dict <python:dict>`
        Data instance for which a prediction will be made. If ``inst`` is of
        type :py:class:`scipy.sparse.csr_matrix`, its first dimension must be 1
        (``shape[0]==1``). If ``inst`` is of type :py:class:`numpy.ndarray`,
        it must be one-dimensional. If ``inst`` is of type
        :py:class:`dict <python:dict>`, it must be a dictionary where the keys
        indicate feature indices (0-based) and the values corresponding
        feature values.
    missing : :py:class:`float <python:float>`, optional
        Value in the data instance that represents a missing value. If set to
        ``None``, ``numpy.nan`` will be used. Only applicable if ``inst`` is
        of type :py:class:`numpy.ndarray`.
    pred_margin: :py:class:`bool <python:bool>`, optional
        Whether to produce raw margins rather than transformed probabilities
    """
    entry = (PredictorEntry * self.num_feature_)()
    for i in range(self.num_feature_):
      entry[i].missing = -1

    if isinstance(inst, scipy.sparse.csr_matrix):
      if inst.shape[0] != 1:
        raise ValueError('inst cannot have more than one row')
      if inst.shape[1] > self.num_feature_:
        raise ValueError('Too many features. This model was trained with only '+\
                         '{} features'.format(self.num_feature_))
      for i in range(inst.nnz):
        entry[inst.indices[i]].fvalue = inst.data[i]
    elif isinstance(inst, scipy.sparse.csc_matrix):
      raise TypeError('inst must be csr_matrix')
    elif isinstance(inst, np.ndarray):
      if len(inst.shape) > 1:
        raise ValueError('inst must be 1D')
      if inst.shape[0] > self.num_feature_:
        raise ValueError('Too many features. This model was trained with only '+\
                         '{} features'.format(self.num_feature_))
      if missing is None or np.isnan(missing):
        for i in range(inst.shape[0]):
          if not np.isnan(inst[i]):
            entry[i].fvalue = inst[i]
      else:
        for i in range(inst.shape[0]):
          if inst[i] != missing:
            entry[i].fvalue = inst[i]
    elif isinstance(inst, dict):
      for k, v in inst.items():
        entry[k].fvalue = v
    else:
      raise TypeError('inst must be NumPy array, SciPy CSR matrix, or a dictionary')

    result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorQueryResultSizeSingleInst(
        self.handle,
        ctypes.byref(result_size)))
    out_result = np.zeros(result_size.value, dtype=np.float32, order='C')
    out_result_size = ctypes.c_size_t()
    _check_call(_LIB.TreelitePredictorPredictInst(
        self.handle,
        ctypes.byref(entry),
        ctypes.c_int(1 if pred_margin else 0),
        out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(out_result_size)))
    idx = int(out_result_size.value)
    res = out_result[0:idx].reshape((1, -1)).squeeze()
    if self.num_output_group_ > 1:
      res = res.reshape((-1, self.num_output_group_))
    return res

  def predict(self, batch, verbose=False, pred_margin=False):
    """
    Perform batch prediction with a 2D sparse data matrix. Worker threads will
    internally divide up work for batch prediction. **Note that this function
    may be called by only one thread at a time.** In order to use multiple
    threads to process multiple prediction requests simultaneously, use
    :py:meth:`predict_instance` instead.

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
    idx = int(out_result_size.value)
    res = out_result[0:idx].reshape((batch.shape()[0], -1)).squeeze()
    if self.num_output_group_ > 1 and batch.shape()[0] != idx:
      res = res.reshape((-1, self.num_output_group_))
    return res

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreelitePredictorFree(self.handle))
      self.handle = None

  @property
  def num_feature(self):
    """Query number of features used in the model"""
    return self.num_feature_

  @property
  def num_output_group(self):
    """Query number of output groups of the model"""
    return self.num_output_group_

  @property
  def pred_transform(self):
    """Query pred transform of the model"""
    return self.pred_transform_

  @property
  def global_bias(self):
    """Query global bias of the model"""
    return self.global_bias_

  @property
  def sigmoid_alpha(self):
    """Query sigmoid alpha of the model"""
    return self.sigmoid_alpha_

__all__ = ['Predictor', 'Batch', '__version__']
