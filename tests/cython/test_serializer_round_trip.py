"""Test for serialization, via buffer protocol"""
import os
import pytest
import treelite
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from serializer import treelite_serialize, treelite_deserialize


@pytest.mark.parametrize("clazz", [RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier])
def test_serialize_as_buffer(clazz):
    """Test whether Treelite objects can be serialized to a buffer"""
    X, y = load_iris(return_X_y=True)
    params = {
        "max_depth": 5,
        "random_state": 0,
        "n_estimators": 10
    }
    if clazz == GradientBoostingClassifier:
        params["init"] = "zero"
    clf = clazz(**params)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

    # Prediction should be correct after a round-trip
    tl_model = treelite.sklearn.import_model(clf)
    frames = treelite_serialize(tl_model)
    tl_model2 = treelite_deserialize(frames)
    out_prob = treelite.gtil.predict(tl_model2, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)

    # The model should serialize to the same byte sequence after a round-trip
    frames2 = treelite_serialize(tl_model2)
    assert frames["header"] == frames2["header"]
    assert len(frames["frames"]) == len(frames2["frames"])
    for x, y in zip(frames["frames"], frames2["frames"]):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("clazz", [RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier])
def test_serialize_as_checkpoint(tmpdir, clazz):
    """Test whether Treelite objects can be serialized to a checkpoint"""
    X, y = load_iris(return_X_y=True)
    params = {
        "max_depth": 5,
        "random_state": 0,
        "n_estimators": 10
    }
    if clazz == GradientBoostingClassifier:
        params["init"] = "zero"
    clf = clazz(**params)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

    # Prediction should be correct after a round-trip
    tl_model = treelite.sklearn.import_model(clf)
    checkpoint_path = os.path.join(tmpdir, "checkpoint.bin")
    tl_model.serialize(checkpoint_path)
    tl_model2 = treelite.Model.deserialize(checkpoint_path)
    out_prob = treelite.gtil.predict(tl_model2, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)

    # The model should serialize to the same byte sequence after a round-trip
    checkpoint_path2 = os.path.join(tmpdir, "checkpoint2.bin")
    tl_model2.serialize(checkpoint_path2)
    with open(checkpoint_path, "rb") as f, open(checkpoint_path2, "rb") as f2:
        checkpoint = f.read()
        checkpoint2 = f2.read()
    assert checkpoint == checkpoint2
