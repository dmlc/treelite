from ..frontend import Model
from ..core import _LIB, _check_call
import ctypes
import numpy as np


def predict(model: Model, input: np.ndarray, pred_margin: bool = False):
    assert isinstance(model, Model)
    assert isinstance(input, np.ndarray)
    assert len(input.shape) == 2
    input = np.array(input, copy=False, dtype=np.float32, order='C')
    output_size = ctypes.c_size_t()
    _check_call(_LIB.TreeliteGTILGetPredictOutputSize(model.handle, ctypes.c_size_t(input.shape[0]),
                                                      ctypes.byref(output_size)))
    out_result = np.zeros(shape=output_size.value, dtype=np.float32, order='C')
    out_result_size = ctypes.c_size_t()
    _check_call(_LIB.TreeliteGTILPredict(
        model.handle,
        input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(input.shape[0]),
        out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(0 if pred_margin else 1),
        ctypes.byref(out_result_size)
    ))
    idx = int(out_result_size.value)
    res = out_result[0:idx].reshape((input.shape[0], -1)).squeeze()
    if model.num_class > 1 and input.shape[0] != idx:
        res = res.reshape((-1, model.num_class))
    return res
