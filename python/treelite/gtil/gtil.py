"""
General Tree Inference Library (GTIL)
"""
import ctypes
import json

import numpy as np

from ..core import _LIB, _check_call
from ..frontend import Model
from ..util import c_str


class GTILConfig:
    """Object holding configuration data"""

    # pylint: disable=too-few-public-methods

    def __init__(self, *, nthread: int, pred_margin: bool):
        predictor_config = {
            "nthread": nthread,
            "predict_type": ("raw" if pred_margin else "default"),
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
    model: Model, data: np.ndarray, nthread: int = -1, pred_margin: bool = False
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
    pred_margin : :py:class:`bool <python:bool>`, optional
        Whether to produce raw margin scores

    Returns
    -------
    prediction : :py:class:`numpy.ndarray` array
        Prediction
    """
    assert isinstance(model, Model)
    assert isinstance(data, np.ndarray)
    assert len(data.shape) == 2
    data = np.array(data, copy=False, dtype=np.float32, order="C")
    output_size = ctypes.c_size_t()
    config = GTILConfig(nthread=nthread, pred_margin=pred_margin)
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
    _check_call(
        _LIB.TreeliteGTILPredictEx(
            model.handle,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(data.shape[0]),
            out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            config.handle,
            ctypes.byref(out_result_size),
        )
    )
    idx = int(out_result_size.value)
    res = out_result[0:idx].reshape((data.shape[0], -1)).squeeze()
    if model.num_class > 1 and data.shape[0] != idx:
        res = res.reshape((-1, model.num_class))
    return res
