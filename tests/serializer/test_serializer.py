"""Test for serialization, via buffer protocol"""

import ctypes
from typing import List

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import treelite
from treelite.core import _LIB, _check_call
from treelite.model import _numpy2pybuffer, _pybuffer2numpy, _TreelitePyBufferFrame
from treelite.util import c_array


def treelite_deserialize(frames: List[np.ndarray]) -> treelite.Model:
    """Serialize model to PyBuffer frames"""
    buffers = [_numpy2pybuffer(frame) for frame in frames]
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteDeserializeModelFromPyBuffer(
            c_array(_TreelitePyBufferFrame, buffers),
            ctypes.c_size_t(len(buffers)),
            ctypes.byref(handle),
        )
    )
    return treelite.Model(handle=handle)


def treelite_serialize(
    model: treelite.Model,
) -> List[np.ndarray]:
    """Deserialize model from PyBuffer frames"""
    frames = ctypes.POINTER(_TreelitePyBufferFrame)()
    n_frames = ctypes.c_size_t()
    _check_call(
        _LIB.TreeliteSerializeModelToPyBuffer(
            model.handle,
            ctypes.byref(frames),
            ctypes.byref(n_frames),
        )
    )
    return [_pybuffer2numpy(frames[i]) for i in range(n_frames.value)]


@pytest.mark.parametrize(
    "clazz", [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
)
def test_serialize_as_buffer(clazz):
    """Test whether Treelite objects can be serialized to a buffer"""
    X, y = load_iris(return_X_y=True)
    params = {"max_depth": 5, "random_state": 0, "n_estimators": 10}
    if clazz == GradientBoostingClassifier:
        params["init"] = "zero"
    clf = clazz(**params)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X).reshape((1, X.shape[0], -1))

    # Prediction should be correct after a round-trip
    tl_model = treelite.sklearn.import_model(clf)
    frames = treelite_serialize(tl_model)
    tl_model2 = treelite_deserialize(frames)
    out_prob = treelite.gtil.predict(tl_model2, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)

    # The model should serialize to the same byte sequence after a round-trip
    frames2 = treelite_serialize(tl_model2)
    assert len(frames) == len(frames2)
    for x, y in zip(frames, frames2):
        assert np.array_equal(x, y)
