"""
General Tree Inference Library (GTIL)
"""

import ctypes
import json
from dataclasses import asdict, dataclass
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix

from ..core import _LIB, _check_call
from ..frontend import Model
from ..util import c_str, typestr_to_ctypes_type, typestr_to_numpy_type


@dataclass
class GTILConfig:
    """Object holding configuration data"""

    predict_type: str
    # Prediction type is one of the following:
    # * "default": Usual prediction method. Sum over trees and apply post-processing.
    # * "raw": Sum over trees, but don't apply post-processing; get raw margin scores
    #          instead.
    # * "leaf_id": Output one (integer) leaf ID per tree.
    # * "score_per_tree": Output one or more margin scores per tree.
    nthread: int = -1
    # <= 0 indicates using all threads

    def to_json(self):
        """Convert configuration object into a JSON string"""
        return json.dumps(asdict(self))

    def __post_init__(self):
        self._handle = ctypes.c_void_p()
        _check_call(
            _LIB.TreeliteGTILParseConfig(
                c_str(self.to_json()), ctypes.byref(self._handle)
            )
        )

    def __del__(self):
        if self._handle is not None:
            _check_call(_LIB.TreeliteGTILDeleteConfig(self._handle))

    @property
    def handle(self):
        """Access the handle to the associated C++ object"""
        return self._handle


def predict(
    model: Model,
    data: Union[np.ndarray, csr_matrix],
    *,
    nthread: int = -1,
    pred_margin: bool = False,
):
    """
    Predict with a Treelite model using the General Tree Inference Library (GTIL).

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` / :py:class:`scipy.sparse.csr_matrix`
        Data matrix, with which to run prediction
    nthread : :py:class:`int <python:int>`, optional
        Number of CPU cores to use in prediction. If <= 0, use all CPU cores.
    pred_margin : bool
        Whether to produce raw margin scores. If pred_margin=True, post-processing
        is no longer applied and raw margin scores are produced.

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction output. Expected dimensions: (num_row, num_target, max(num_class))
    """
    predict_type = "raw" if pred_margin else "default"

    config = GTILConfig(nthread=nthread, predict_type=predict_type)
    return _predict_impl(model, data, config=config)


def predict_leaf(
    model: Model,
    data: Union[np.ndarray, csr_matrix],
    *,
    nthread: int = -1,
):
    """
    Predict with a Treelite model, outputting the leaf node's ID for each row.

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` / :py:class:`scipy.sparse.csr_matrix`
        Data matrix, with which to run prediction
    nthread : :py:class:`int <python:int>`, optional
        Number of CPU cores to use in prediction. If <= 0, use all CPU cores.

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction output. Expected output dimensions: (num_row, num_tree)

    Notes
    -----
    Treelite assigns a unique integer ID for every node in the tree, including leaf
    nodes as well as internal nodes. It does so by traversing the tree breadth-first.
    So, for example, the root node is assigned ID 0, and the two nodes at depth=1 is
    assigned ID 1 and 2, respectively.
    Call :py:meth:`treelite.Model.dump_as_json` to obtain the ID of every tree node.
    """

    config = GTILConfig(nthread=nthread, predict_type="leaf_id")
    return _predict_impl(model, data, config=config)


def predict_per_tree(
    model: Model, data: Union[np.ndarray, csr_matrix], *, nthread: int = -1
):
    """
    Predict with a Treelite model and output prediction of each tree.
    This function computes one or more margin scores per tree.

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` / :py:class:`scipy.sparse.csr_matrix`
        Data matrix, with which to run prediction
    nthread : :py:class:`int <python:int>`, optional
        Number of CPU cores to use in prediction. If <= 0, use all CPU cores.

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction output. Expected output dimensions:
        (num_row, num_tree, leaf_vector_shape[0] * leaf_vector_shape[1])
    """

    config = GTILConfig(nthread=nthread, predict_type="score_per_tree")
    return _predict_impl(model, data, config=config)


def _predict_impl(
    model: Model, data: Union[np.ndarray, csr_matrix], *, config: GTILConfig
) -> np.ndarray:
    # Validate parameters
    if not isinstance(model, Model):
        raise ValueError('Argument "model" must be a Model type')
    if not isinstance(data, (np.ndarray, csr_matrix)):
        raise ValueError(
            'Argument "data" must be either 2D NumPy array or Scipy CSR matrix'
        )
    if len(data.shape) != 2:
        raise ValueError('Argument "data" must be a 2D array')

    is_dense = isinstance(data, np.ndarray)

    if is_dense:
        data = np.asarray(
            data, dtype=typestr_to_numpy_type(model.input_type), order="C"
        )
        if data.shape[1] < model.num_feature:
            # Pad missing features with NAs
            data = np.pad(
                data,
                ((0, 0), (0, model.num_feature - data.shape[1])),
                "constant",
                constant_values="nan",
            )
            assert data.shape[1] == model.num_feature
    else:
        assert isinstance(data, csr_matrix)
        elems = np.asarray(
            data.data,
            dtype=typestr_to_numpy_type(model.input_type),
            order="C",
        )
        col_ind = np.asarray(data.indices, dtype=np.uint64, order="C")
        row_ptr = np.asarray(data.indptr, dtype=np.uint64, order="C")
    output_shape_ptr = ctypes.POINTER(ctypes.c_uint64)()
    output_ndim = ctypes.c_uint64()
    _check_call(
        _LIB.TreeliteGTILGetOutputShape(
            model.handle,
            ctypes.c_uint64(data.shape[0]),
            config.handle,
            ctypes.byref(output_shape_ptr),
            ctypes.byref(output_ndim),
        )
    )
    output_shape = np.ctypeslib.as_array(output_shape_ptr, shape=(output_ndim.value,))
    out_result = np.zeros(
        shape=output_shape, dtype=typestr_to_numpy_type(model.output_type), order="C"
    )

    if is_dense:
        _check_call(
            _LIB.TreeliteGTILPredict(
                model.handle,
                data.ctypes.data_as(
                    ctypes.POINTER(typestr_to_ctypes_type(model.input_type))
                ),
                c_str(model.input_type),
                ctypes.c_uint64(data.shape[0]),
                out_result.ctypes.data_as(
                    ctypes.POINTER(typestr_to_ctypes_type(model.output_type))
                ),
                config.handle,
            )
        )
    else:
        _check_call(
            _LIB.TreeliteGTILPredictSparse(
                model.handle,
                elems.ctypes.data_as(
                    ctypes.POINTER(typestr_to_ctypes_type(model.input_type))
                ),
                c_str(model.input_type),
                col_ind.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                ctypes.c_uint64(data.shape[0]),
                out_result.ctypes.data_as(
                    ctypes.POINTER(typestr_to_ctypes_type(model.output_type))
                ),
                config.handle,
            )
        )
    return out_result
