"""
General Tree Inference Library (GTIL)
"""
import ctypes
import json
import warnings
from typing import Optional

import numpy as np

from ..core import _LIB, _check_call
from ..frontend import Model
from ..util import c_str


class GTILConfig:
    """Object holding configuration data"""

    # pylint: disable=too-few-public-methods

    def __init__(self, *, nthread: int, predict_type: str):
        predictor_config = {
            "nthread": nthread,
            "predict_type": predict_type,
        }
        predictor_config = json.dumps(predictor_config)
        self.handle = ctypes.c_void_p()
        _check_call(
            _LIB.TreeliteGTILParseConfig(
                c_str(predictor_config), ctypes.byref(self.handle)
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
        pred_margin: Optional[bool] = None,
        predict_type: Optional[str] = None
):
    """
    Predict with a Treelite model using General Tree Inference Library (GTIL).

    .. note:: GTIL is currently experimental

        GTIL is currently in its early stage of development and may have bugs and performance
        issues. Please report any issues found on GitHub.

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    data : :py:class:`numpy.ndarray` array
        2D NumPy array, with which to run prediction
    nthread : :py:class:`int <python:int>`, optional
        Number of CPU cores to use in prediction. If <= 0, use all CPU cores.
    pred_margin : bool, optional
        Deprecated. Set predict_type="raw" instead.
    predict_type : string, optional
        One of the following:
        * "default": Usual prediction method. Sum over trees and apply post-processing.
        * "raw": Sum over trees, but don't apply post-processing; get raw margin scores
                 instead.
        * "leaf_id": Output one (integer) leaf ID per tree.
        * "score_per_tree": Output one or more margin scores per tree.

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction
    """
    # Parameter validation
    if pred_margin is not None:
        if predict_type:
            raise ValueError(
                "Cannot specify pred_margin and predict_type at the same "
                "time. Please set predict_type only."
            )
        warnings.warn(
            "The pred_margin argument is deprecated. Please use "
            'predict_type="raw" instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        predict_type = "raw"
    if not predict_type:
        predict_type = "default"
    if not isinstance(model, Model):
        raise ValueError('Argument "model" must be a Model type')
    if (not isinstance(data, np.ndarray)) or len(data.shape) != 2:
        raise ValueError('Argument "data" must be a 2D NumPy array')

    data = np.array(data, copy=False, dtype=np.float32, order="C")
    output_size = ctypes.c_size_t()
    config = GTILConfig(nthread=nthread, predict_type=predict_type)
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
            ctypes.byref(out_result_shape)
        )
    )
    # Reshape the result according to out_result_shape
    out_shape = np.ctypeslib.as_array(out_result_shape, shape=(out_result_ndim.value,))
    idx = int(out_result_size.value)
    assert idx == np.prod(out_shape)
    res = out_result[0:idx].reshape(out_shape).squeeze()
    return res
