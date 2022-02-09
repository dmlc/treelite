"""
General Tree Inference Library (GTIL)
"""
import ctypes
import numpy as np
from ..frontend import Model
from ..core import _LIB, _check_call


class Predictor:
    """
    Predictor class to perform prediction with a Treelite model.

    General Tree Inference Library (GTIL) is intended to be a reference implementation. GTIL is also
    useful in situations where using a C compiler is not feasible.

    .. note:: GTIL is currently experimental

        GTIL is currently in its early stage of development and may have bugs and performance
        issues. Please report any issues found on GitHub.

    Parameters
    ----------
    model : :py:class:`Model` object
        Treelite model object
    """
    def __init__(self, model: Model):
        assert isinstance(model, Model)
        handle = ctypes.c_void_p()
        _check_call(_LIB.TreeliteGTILCreatePredictor(model.handle, ctypes.byref(handle)))
        self.handle = handle
        self.num_class = model.num_class

    def __del__(self):
        if self.handle is not None:
            _check_call(_LIB.TreeliteGTILDeletePredictor(self.handle))
            self.handle = None

    def predict(self, data: np.ndarray, nthread: int = -1, pred_margin: bool = False):
        """
        Predict with a 2D NumPy array.

        Parameters
        ----------
        data : :py:class:`Model` object
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
        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 2
        data = np.array(data, copy=False, dtype=np.float32, order='C')
        output_size = ctypes.c_size_t()
        _check_call(_LIB.TreeliteGTILPredictorQueryResultSize(self.handle,
                                                              ctypes.c_size_t(data.shape[0]),
                                                              ctypes.byref(output_size)))
        out_result = np.zeros(shape=output_size.value, dtype=np.float32, order='C')
        out_result_size = ctypes.c_size_t()
        _check_call(_LIB.TreeliteGTILPredictorPredict(
            self.handle,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(data.shape[0]),
            out_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(nthread),
            ctypes.c_int(0 if pred_margin else 1),
            ctypes.byref(out_result_size)
        ))
        idx = int(out_result_size.value)
        res = out_result[0:idx].reshape((data.shape[0], -1)).squeeze()
        if self.num_class > 1 and data.shape[0] != idx:
            res = res.reshape((-1, self.num_class))
        return res
