# coding: utf-8
# pylint: disable=W0201
"""predictor module"""
import ctypes
import sys
import os
import re
import pathlib
import numpy as np
import scipy.sparse
from .util import c_str, py_str, _log_callback, TreeliteRuntimeError, lineno, log_info, \
    lib_extension_current_platform, type_info_to_ctypes_type, type_info_to_numpy_type, \
    numpy_type_to_type_info
from .libpath import TreeliteRuntimeLibraryNotFound, find_lib_path


def _load_runtime_lib():
    """Load Treelite runtime"""
    lib_path = find_lib_path()
    if sys.version_info >= (3, 8) and sys.platform == 'win32':
        # pylint: disable=no-member
        os.add_dll_directory(os.path.join(os.path.normpath(sys.prefix), 'Library', 'bin'))
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    lib.TreeliteGetLastError.restype = ctypes.c_char_p
    lib.callback = _log_callback
    if lib.TreeliteRegisterLogCallback(lib.callback) != 0:
        raise TreeliteRuntimeError(py_str(lib.TreeliteGetLastError()))
    return lib


# load the Treelite runtime
# (do not load if called by sphinx)
if 'sphinx' in sys.modules:
    try:
        _LIB = _load_runtime_lib()
    except TreeliteRuntimeLibraryNotFound:
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
        raise TreeliteRuntimeError(py_str(_LIB.TreeliteGetLastError()))


class Predictor:
    """
    Predictor class: loader for compiled shared libraries

    Note:
        Treelite uses a custom thread pool which pins threads to CPU cores by default.
        To disable thread pinning, set the environment variable
        ``TREELITE_BIND_THREADS`` to ``0``. Disabling thread pinning is recommended when
        using Treelite in multi-threaded applications.

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
            lib_found = False
            dir = pathlib.Path(libpath)
            ext = lib_extension_current_platform()
            for candidate in dir.glob(f'*{ext}'):
                try:
                    path = str(candidate.resolve(strict=True))
                    lib_found = True
                    break
                except FileNotFoundError:
                    continue
            if not lib_found:
                raise TreeliteRuntimeError(f'Directory {libpath} doesn\'t appear ' +
                                           'to have any dynamic shared library (.so/.dll/.dylib).')
        else:  # libpath is actually the name of shared library file
            fileext = os.path.splitext(libpath)[1]
            if fileext == '.dll' or fileext == '.so' or fileext == '.dylib':
                path = libpath
            else:
                raise TreeliteRuntimeError(f'Specified path {libpath} has wrong file extension ' +
                                           f'({fileext}); the share library must have one of the ' +
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
        # save # of classes
        num_class = ctypes.c_size_t()
        _check_call(_LIB.TreelitePredictorQueryNumClass(
            self.handle,
            ctypes.byref(num_class)))
        self.num_class_ = num_class.value
        # save # of pred transform
        pred_transform = ctypes.c_char_p()
        _check_call(_LIB.TreelitePredictorQueryPredTransform(
            self.handle,
            ctypes.byref(pred_transform)))
        self.pred_transform_ = py_str(pred_transform.value)
        # save # of sigmoid alpha
        sigmoid_alpha = ctypes.c_float()
        _check_call(_LIB.TreelitePredictorQuerySigmoidAlpha(
            self.handle,
            ctypes.byref(sigmoid_alpha)))
        self.sigmoid_alpha_ = sigmoid_alpha.value
        # save # of ratio c
        ratio_c = ctypes.c_float()
        _check_call(_LIB.TreelitePredictorQueryRatioC(
            self.handle,
            ctypes.byref(ratio_c)))
        self.ratio_c_ = ratio_c.value
        # save # of global bias
        global_bias = ctypes.c_float()
        _check_call(_LIB.TreelitePredictorQueryGlobalBias(
            self.handle,
            ctypes.byref(global_bias)))
        self.global_bias_ = global_bias.value
        threshold_type = ctypes.c_char_p()
        _check_call(_LIB.TreelitePredictorQueryThresholdType(
            self.handle,
            ctypes.byref(threshold_type)))
        self.threshold_type_ = py_str(threshold_type.value)
        leaf_output_type = ctypes.c_char_p()
        _check_call(_LIB.TreelitePredictorQueryLeafOutputType(
            self.handle,
            ctypes.byref(leaf_output_type)))
        self.leaf_output_type_ = py_str(leaf_output_type.value)

        if verbose:
            log_info(__file__, lineno(),
                     f'Dynamic shared library {path} has been successfully loaded into memory')

    def predict(self, dmat, verbose=False, pred_margin=False):
        """
        Perform batch prediction with a 2D sparse data matrix. Worker threads will
        internally divide up work for batch prediction. **Note that this function
        may be called by only one thread at a time.**

        Parameters
        ----------
        dmat: object of type :py:class:`DMatrix`
            batch of rows for which predictions will be made
        verbose : :py:class:`bool <python:bool>`, optional
            Whether to print extra messages during prediction
        pred_margin: :py:class:`bool <python:bool>`, optional
            whether to produce raw margins rather than transformed probabilities
        """
        if not isinstance(dmat, DMatrix):
            raise TreeliteRuntimeError('dmat must be of type DMatrix')
        result_size = ctypes.c_size_t()
        _check_call(_LIB.TreelitePredictorQueryResultSize(
            self.handle,
            dmat.handle,
            ctypes.byref(result_size)))
        result_type = ctypes.c_char_p()
        _check_call(_LIB.TreelitePredictorQueryLeafOutputType(
            self.handle,
            ctypes.byref(result_type)))
        result_type = py_str(result_type.value)
        out_result = np.zeros(result_size.value,
                              dtype=type_info_to_numpy_type(result_type),
                              order='C')
        out_result_size = ctypes.c_size_t()
        _check_call(_LIB.TreelitePredictorPredictBatch(
            self.handle,
            dmat.handle,
            ctypes.c_int(1 if verbose else 0),
            ctypes.c_int(1 if pred_margin else 0),
            out_result.ctypes.data_as(ctypes.POINTER(type_info_to_ctypes_type(result_type))),
            ctypes.byref(out_result_size)))
        idx = int(out_result_size.value)
        res = out_result[0:idx].reshape((dmat.shape[0], -1)).squeeze()
        if self.num_class_ > 1 and dmat.shape[0] != idx:
            res = res.reshape((-1, self.num_class_))
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
    def num_class(self):
        """Query number of output groups of the model"""
        return self.num_class_

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

    @property
    def ratio_c(self):
        """Query sigmoid alpha of the model"""
        return self.ratio_c_

    @property
    def threshold_type(self):
        """Query threshold type of the model"""
        return self.threshold_type_

    @property
    def leaf_output_type(self):
        """Query threshold type of the model"""
        return self.leaf_output_type_

class DMatrix:
    """Data matrix used in Treelite.

    Parameters
    ----------
    data : :py:class:`str <python:str>` / :py:class:`numpy.ndarray` /\
           :py:class:`scipy.sparse.csr_matrix` / :py:class:`pandas.DataFrame`
        Data source. When data is :py:class:`str <python:str>` type, it indicates
        that data should be read from a file.
    data_format : :py:class:`str <python:str>`, optional
        Format of input data file. Applicable only when data is read from a
        file. If missing, the svmlight (.libsvm) format is assumed.
    dtype : :py:class:`str <python:str>`, optional
        If specified, the data will be casted into the corresponding data type.
    missing : :py:class:`float <python:float>`, optional
        Value in the data that represents a missing entry. If set to ``None``,
        ``numpy.nan`` will be used.
    verbose : :py:class:`bool <python:bool>`, optional
        Whether to print extra messages during construction
    feature_names : :py:class:`list <python:list>`, optional
        Human-readable names for features
    feature_types : :py:class:`list <python:list>`, optional
        Types for features
    nthread : :py:class:`int <python:int>`, optional
        Number of threads
    """

    # pylint: disable=R0902,R0903,R0913

    def __init__(self, data, data_format=None, dtype=None, missing=None,
                 feature_names=None, feature_types=None,
                 verbose=False, nthread=None):
        if data is None:
            raise TreeliteRuntimeError("'data' argument cannot be None")

        self.handle = ctypes.c_void_p()

        if isinstance(data, (str,)):
            raise TreeliteRuntimeError(
                "'data' argument cannot be a string. Did you mean to load data from a text file? "
                "Please use the following packages to load the text file:\n"
                "   * CSV file: Use pandas.read_csv() or numpy.loadtxt()\n"
                "   * LIBSVM file: Use sklearn.datasets.load_svmlight_file()")
        elif isinstance(data, scipy.sparse.csr_matrix):
            self._init_from_csr(data, dtype=dtype)
        elif isinstance(data, scipy.sparse.csc_matrix):
            self._init_from_csr(data.tocsr(), dtype=dtype)
        elif isinstance(data, np.ndarray):
            self._init_from_npy2d(data, missing, dtype=dtype)
        else:  # any type that's convertible to CSR matrix is O.K.
            try:
                csr = scipy.sparse.csr_matrix(data)
                self._init_from_csr(csr, dtype=dtype)
            except Exception as e:
                raise TypeError(f'Cannot initialize DMatrix from {type(data).__name__}') from e
        self.feature_names = feature_names
        self.feature_types = feature_types
        num_row, num_col, nelem = self._get_dims()
        self.shape = (num_row, num_col)
        self.size = nelem

    def _init_from_csr(self, csr, dtype=None):
        """Initialize data from a CSR (Compressed Sparse Row) matrix"""
        if len(csr.indices) != len(csr.data):
            raise ValueError('indices and data not of same length: {} vs {}'
                             .format(len(csr.indices), len(csr.data)))
        if len(csr.indptr) != csr.shape[0] + 1:
            raise ValueError('len(indptr) must be equal to 1 + [number of rows]' \
                             + 'len(indptr) = {} vs 1 + [number of rows] = {}'
                             .format(len(csr.indptr), 1 + csr.shape[0]))
        if csr.indptr[-1] != len(csr.data):
            raise ValueError('last entry of indptr must be equal to len(data)' \
                             + 'indptr[-1] = {} vs len(data) = {}'
                             .format(csr.indptr[-1], len(csr.data)))

        if dtype is None:
            data_type = csr.data.dtype
        else:
            data_type = type_info_to_numpy_type(dtype)
        data_type_code = numpy_type_to_type_info(data_type)
        data_ptr_type = ctypes.POINTER(type_info_to_ctypes_type(data_type_code))
        if data_type_code not in ['float32', 'float64']:
            raise ValueError('data should be either float32 or float64 type')

        data = np.array(csr.data, copy=False, dtype=data_type, order='C')
        indices = np.array(csr.indices, copy=False, dtype=np.uintc, order='C')
        indptr = np.array(csr.indptr, copy=False, dtype=np.uintp, order='C')
        _check_call(_LIB.TreeliteDMatrixCreateFromCSR(
            data.ctypes.data_as(data_ptr_type),
            c_str(data_type_code),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            ctypes.c_size_t(csr.shape[0]),
            ctypes.c_size_t(csr.shape[1]),
            ctypes.byref(self.handle)))

    def _init_from_npy2d(self, mat, missing, dtype=None):
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
        if dtype is None:
            data_type = mat.dtype
        else:
            data_type = type_info_to_numpy_type(dtype)
        data_type_code = numpy_type_to_type_info(data_type)
        data_ptr_type = ctypes.POINTER(type_info_to_ctypes_type(data_type_code))
        if data_type_code not in ['float32', 'float64']:
            raise ValueError('data should be either float32 or float64 type')
        # flatten the array by rows and ensure it is float32.
        # we try to avoid data copies if possible
        # (reshape returns a view when possible and we explicitly tell np.array to
        #  avoid copying)
        data = np.array(mat.reshape(mat.size), copy=False, dtype=data_type)
        missing = missing if missing is not None else np.nan
        missing = np.array([missing], dtype=data_type, order='C')
        _check_call(_LIB.TreeliteDMatrixCreateFromMat(
            data.ctypes.data_as(data_ptr_type),
            c_str(data_type_code),
            ctypes.c_size_t(mat.shape[0]),
            ctypes.c_size_t(mat.shape[1]),
            missing.ctypes.data_as(data_ptr_type),
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
        if self.handle:
            _check_call(_LIB.TreeliteDMatrixFree(self.handle))
            self.handle = None

    def __repr__(self):
        return '<{}x{} sparse matrix of type treelite.DMatrix\n' \
                   .format(self.shape[0], self.shape[1]) \
               + '        with {} stored elements in Compressed Sparse Row format>' \
                   .format(self.size)


__all__ = ['Predictor', 'DMatrix']
