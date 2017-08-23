# coding: utf-8
"""Core tree-lite library."""
from __future__ import absolute_import

import sys
import os
import ctypes
import collections

import numpy as np
import scipy.sparse

from .libpath import find_lib_path
from .compat import STRING_TYPES, PY3, DataFrame, py_str, PANDAS_INSTALLED

class TreeliteError(Exception):
    """Error thrown by tree-lite"""
    pass

def _load_lib():
    """Load tree-lite Library."""
    lib_path = find_lib_path()
    if len(lib_path) == 0:
        return None
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.TreeliteGetLastError.restype = ctypes.c_char_p
    return lib

# load the tree-lite library globally
_LIB = _load_lib()

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

def c_str(string):
  """Convert a Python string to C string"""
  return ctypes.c_char_p(string.encode('utf-8'))

def c_array(ctype, values):
  """Convert a Python byte array to C array"""
  return (ctype * len(values))(*values)

# import frontend later to break circular reference
from .frontend import Model, load_model_from_file

PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int',
                       'int64': 'int', 'uint8': 'int', 'uint16': 'int',
                       'uint32': 'int', 'uint64': 'int', 'float16': 'float',
                       'float32': 'float', 'float64': 'float', 'bool': 'i'}

def _maybe_pandas_data(data, feature_names, feature_types):
  """Extract internal data from pd.DataFrame for DMatrix data"""
  if not isinstance(data, DataFrame):
    return data, feature_names, feature_types
  data_dtypes = data.dtypes
  if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
    bad_fields = [data.columns[i] for i, dtype in
               enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]
    msg = "DataFrame.dtypes for data must be in, float, or bool. Did not " \
          + "expect the data types in fields "
    raise ValueError(msg + ', '.join(bad_fields))
  if feature_names is None:
    feature_names = data.columns.format()
  if feature_types is None:
    feature_types = [PANDAS_DTYPE_MAPPER[dtype.name] for dtype in data_dtypes]
  data = data.values.astype('float')
  return data, feature_names, feature_types

class DMatrix(object):
  """Data matrix used in tree-lite."""
  
  def __init__(self, data, data_format=None, missing=None,
               feature_names=None, feature_types=None,
               verbose=False, nthread=None):
    """Data matrix used in tree-lite.

    Parameters
    ----------
    data : string/numpy array/scipy.sparse/pd.DataFrame
        Data source of DMatrix.
        When data is string type, it indicates that data should be read from
        a file.
    data_format: string, optional
        Format of input data file. Applicable only when data is read from a
        file. When unspecified, the svmlight (*.libsvm) format is assumed.
    missing : float, optional
        Value in the data that represents a missing entry. If None, defaults to
        np.nan.
    verbose : boolean, optional
        Whether to print extra messages during construction
    feature_names : list, optional
        Human-readable names for features
    feature_types : list, optional
        Types for features
    nthread : integer, optional
        Number of threads
    """
    if data is None:  # empty DMatrix
      self.handle = None
      return

    data, feature_names, feature_types = _maybe_pandas_data(data,
                                                            feature_names,
                                                            feature_types)
    if isinstance(data, STRING_TYPES):
      self.handle = ctypes.c_void_p()
      nthread = nthread if nthread is not None else 0
      data_format = data_format if data_format is not None else "libsvm"
      _check_call(_LIB.TreeliteDMatrixCreateFromFile(c_str(data),
                                                     c_str(data_format),
                                                     ctypes.c_int(nthread),
                                                     ctypes.c_int(verbose),
                                                    ctypes.byref(self.handle)))
    elif isinstance(data, scipy.sparse.csr_matrix):
      self._init_from_csr(data)
    elif isinstance(data, scipy.sparse.csc_matrix):
      self._init_from_csr(data.tocsr())
    elif isinstance(data, np.ndarray):
      self._init_from_npy2d(data, missing)
    else:  # any type that's convertible to CSR matrix is O.K.
      try:
        csr = scipy.sparse.csr_matrix(data)
        self._init_from_csr(csr)
      except:
        raise TypeError('Cannot initialize DMatrix from {}'
                        .format(type(data).__name__))
    self.feature_names = feature_names
    self.feature_types = feature_types

  def shape(self):
    """
    Get dimensions of the DMatrix

    Returns
    -------
    tuple (number of rows, number of columns)
    """
    num_row, num_col, _ = self._get_dims()
    return (num_row, num_col)

  def nnz(self):
    """
    Get number of nonzero entries in the DMatrix

    Returns
    -------
    number of nonzero entries
    """
    _, _, nelem = self._get_dims()
    return nelem

  def _init_from_csr(self, csr):
    """Initialize data from a CSR (Compressed Sparse Row) matrix"""
    if len(csr.indices) != len(csr.data):
      raise ValueError('indices and data not of same length: {} vs {}'
                        .format(len(csr.indices), len(csr.data)))
    if len(csr.indptr) != csr.shape[0] + 1:
      raise ValueError('len(indptr) must be equal to 1 + [number of rows]' \
                        + 'len(indptr) = {} vs 1 + [number of rows] = {}'
                          .format(len(indptr), 1 + csr.shape[0]))
    if csr.indptr[-1] != len(csr.data):
      raise ValueError('last entry of indptr must be equal to len(data)' \
                        + 'indptr[-1] = {} vs len(data) = {}'
                          .format(indptr[-1], len(data)))
    self.handle = ctypes.c_void_p()
    _check_call(
      _LIB.TreeliteDMatrixCreateFromCSR(c_array(ctypes.c_float, csr.data),
                                        c_array(ctypes.c_uint, csr.indices),
                                        c_array(ctypes.c_size_t, csr.indptr),
                                        ctypes.c_size_t(csr.shape[0]),
                                        ctypes.c_size_t(csr.shape[1]),
                                        ctypes.byref(self.handle)))

  def _init_from_npy2d(self, mat, missing):
    """
    Initialize data from a 2-D numpy matrix.
    If ``mat`` does not have ``order='C'`` (also known as row-major) or is not
    contiguous, a temporary copy will be made.
    If ``mat`` does not have ``dtype=numpy.float32``, a temporary copy will be
    made also.
    Thus, as many as two temporary copies of data can be made. One should set
    input layout and type judiciously to conserve memory.
    """
    if len(mat.shape) != 2:
      raise ValueError('Input numpy.ndarray must be two-dimensional')
    # flatten the array by rows and ensure it is float32.
    # we try to avoid data copies if possible
    # (reshape returns a view when possible and we explicitly tell np.array to
    #  avoid copying)
    data = np.array(mat.reshape(mat.size), copy=False, dtype=np.float32)
    self.handle = ctypes.c_void_p()
    missing = missing if missing is not None else np.nan
    _check_call(_LIB.TreeliteDMatrixCreateFromMat(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(mat.shape[0]),
        ctypes.c_size_t(mat.shape[1]),
        ctypes.c_float(missing),
        ctypes.byref(self.handle)))

    
  def _get_dims(self):
    num_row = ctypes.c_size_t()
    num_col = ctypes.c_size_t()
    nelem = ctypes.c_size_t()
    _check_call(_LIB.TreeliteDMatrixGetDimension(self.handle,
                                                  ctypes.byref(num_row),
                                                  ctypes.byref(num_col),
                                                  ctypes.byref(nelem)))
    return (num_row.value, num_col.value, nelem.value)

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreeliteDMatrixFree(self.handle))
      self.handle = None

  def __repr__(self):
    num_row, num_col, nelem = self._get_dims()
    return '<{}x{} sparse matrix of type treelite.DMatrix\n'\
           .format(num_row, num_col) \
        + '        with {} stored elements in Compressed Sparse Row format>'\
           .format(nelem)
  
  def __str__(self):
    # Print first and last 25 non-zero entries
    preview = ctypes.c_char_p()
    _check_call(_LIB.TreeliteDMatrixGetPreview(self.handle,
                                               ctypes.byref(preview)))
    return py_str(preview.value)

class Compiler(object):
  """
  Compiler object to translate a tree ensemble model into a semantic model
  """
  
  def __init__(self, name):
    """Compiler object to translate a tree ensemble model into a semantic model

    Parameters
    ----------
    name : string
        name of compiler
    """
    self.handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteCompilerCreate(c_str(name),
                                            ctypes.byref(self.handle)))

  def generate_code(self, model, path_prefix, params, verbose=False):
    """
    generate prediction code from a tree ensemble model. The code will be C99
    compliant. One header file (.h) will be generated, along with one or more
    source files (.c).

    Usage
    -----
    compiler.generate_code(model, path_prefix="./my/model", verbose=True);
    // files to generate: ./my/model.h, ./my/model.c
    // if parallel compilation is enabled:
    // ./my/model.h, ./my/model0.c, ./my/model1.c, ./my/model2.c, and so forth

    Parameters
    ----------
    model : `Model`
      decision tree ensemble model
    path_prefix : string
      path prefix for header and source files
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
                                                  c_str(path_prefix)))

  def __del__(self):
    if self.handle is not None:
      _check_call(_LIB.TreeliteCompilerFree(self.handle))
      self.handle = None

  def _set_param(self, params, value=None):
    """
    Set parameters into the Booster.

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
                                                c_str(STRING_TYPES[0](val))))
