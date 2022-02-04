"""
General Tree Inference Library (GTIL)
"""
import ctypes
import numpy as np
from ..frontend import Model
from ..core import _LIB, _check_call


def predict(model: Model, data: np.ndarray, nthread: int = -1, pred_margin: bool = False):
    """
    Predict with a Treelite model using General Tree Inference Library (GTIL). GTIL is intended to
    be a reference implementation. GTIL is also useful in situations where using a C compiler is
    not feasible.

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
    data = np.array(data, copy=False, dtype=np.float32, order='C')
    output_size = ctypes.c_size_t()
    _check_call(_LIB.TreeliteGTILGetPredictOutputSize(model.handle, ctypes.c_size_t(data.shape[0]),
                                                      ctypes.byref(output_size)))
    out_result = np.zeros(shape=output_size.value, dtype=np.float32, order='C')
    out_result_size = ctypes.c_size_t()
    _check_call(_LIB.TreeliteGTILPredict(
        model.handle,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(data.shape[0]),
        out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(nthread),
        ctypes.c_int(0 if pred_margin else 1),
        ctypes.byref(out_result_size)
    ))
    idx = int(out_result_size.value)
    res = out_result[0:idx].reshape((data.shape[0], -1)).squeeze()
    if model.num_class > 1 and data.shape[0] != idx:
        res = res.reshape((-1, model.num_class))
    return res
