"""
General Tree Inference Library (GTIL)
"""
import ctypes
import json
import warnings
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np

from ..core import _LIB, _check_call
from ..frontend import Model
from ..util import c_str


@dataclass
class GTILConfig:
    """Object holding configuration data"""

    predict_type: Literal["default", "raw", "leaf_id", "score_per_tree"]
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
        self.handle = ctypes.c_void_p()
        _check_call(
            _LIB.TreeliteGTILParseConfig(
                c_str(self.to_json()), ctypes.byref(self.handle)
            )
        )

    def __del__(self):
        if self.handle is not None:
            _check_call(_LIB.TreeliteGTILDeleteConfig(self.handle))


def predict(
    model: Model,
    data: np.ndarray,
    *,
    nthread: int = -1,
    pred_margin: Optional[bool] = None
):
    """
    Predict with a Treelite model using the General Tree Inference Library (GTIL).

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` array
        2D NumPy array, with which to run prediction
    nthread : :py:class:`int <python:int>`, optional
        Number of CPU cores to use in prediction. If <= 0, use all CPU cores.
    pred_margin : bool, optional (defaults to False)
        Whether to produce raw margin scores. If pred_margin=True, post-processing
        is no longer applied and raw margin scores are produced.

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction output. Expected output dimensions:

        - (num_row,) for regressors and binary classifiers
        - (num_row, num_class) for multi-class classifiers (See Notes for a special
          case.)

    Notes
    -----
    The output has shape (num_row,) if the model is a multi-class classifier with
    task_type="MultiClfGrovePerClass" and pred_transform="max_index".
    """
    if pred_margin is None:
        pred_margin = False
    predict_type = "raw" if pred_margin else "default"

    config = GTILConfig(nthread=nthread, predict_type=predict_type)
    return _predict_impl(model, data, config=config)


def predict_leaf(model: Model, data: np.ndarray, *, nthread: int = -1):
    """
    Predict with a Treelite model, outputting the leaf node's ID for each row.

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` array
        2D NumPy array, with which to run prediction
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


def predict_per_tree(model: Model, data: np.ndarray, *, nthread: int = -1):
    """
    Predict with a Treelite model and output prediction of each tree.
    This function computes one or more margin scores per tree.

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` array
        2D NumPy array, with which to run prediction
    nthread : :py:class:`int <python:int>`, optional
        Number of CPU cores to use in prediction. If <= 0, use all CPU cores.

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction output. Expected output dimensions:

        - (num_row, num_tree) for regressors, binary classifiers,
          and multi-class classifiers with task_type="MultiClfGrovePerClass"
        - (num_row, num_tree, num_class) for multi-class classifiers with
          task_type="kMultiClfProbDistLeaf"
    """

    config = GTILConfig(nthread=nthread, predict_type="score_per_tree")
    return _predict_impl(model, data, config=config)


def _predict_impl(model, data, *, config):
    # Validate parameters
    if not isinstance(model, Model):
        raise ValueError('Argument "model" must be a Model type')
    if (not isinstance(data, np.ndarray)) or len(data.shape) != 2:
        raise ValueError('Argument "data" must be a 2D NumPy array')

    if data.dtype != np.float32:
        warnings.warn(
            "GTIL currently only supports float32 type; data will be "
            "converted to float32 and information might be lost.",
            UserWarning,
        )
    data = np.array(data, copy=False, dtype=np.float32, order="C")
    output_size = ctypes.c_size_t()

    _check_call(
        _LIB.TreeliteGTILGetPredictOutputSizeEx(
            model.handle,
            ctypes.c_size_t(data.shape[0]),
            config.handle,
            ctypes.byref(output_size),
        )
    )
    out_result = np.zeros(shape=output_size.value, dtype=np.float32, order="C")
    out_result_size = ctypes.c_size_t()
    out_result_ndim = ctypes.c_size_t()
    out_result_shape = ctypes.POINTER(ctypes.c_size_t)()
    _check_call(
        _LIB.TreeliteGTILPredictEx(
            model.handle,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(data.shape[0]),
            out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            config.handle,
            ctypes.byref(out_result_size),
            ctypes.byref(out_result_ndim),
            ctypes.byref(out_result_shape),
        )
    )
    # Reshape the result according to out_result_shape
    out_shape = np.ctypeslib.as_array(out_result_shape, shape=(out_result_ndim.value,))
    idx = int(out_result_size.value)
    assert idx == np.prod(out_shape)
    res = out_result[0:idx].reshape(out_shape)
    if data.shape[0] > 1:
        res = res.squeeze()
    return res
