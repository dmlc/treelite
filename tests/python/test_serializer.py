"""Test for serializer"""

import pathlib

import numpy as np
import pytest

import treelite

try:
    from hypothesis import given, settings
    from hypothesis.strategies import data as hypothesis_callback
    from hypothesis.strategies import floats, integers, just, sampled_from
except ImportError:
    pytest.skip("hypothesis not installed; skipping", allow_module_level=True)

try:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
except ImportError:
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)
from .util import TemporaryDirectory


@given(
    clazz=sampled_from(
        [
            RandomForestClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
        ]
    ),
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=2, max_value=5),
    ),
    n_estimators=integers(min_value=3, max_value=10),
    max_depth=integers(min_value=3, max_value=10),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_serialize_as_bytes(clazz, dataset, n_estimators, max_depth, callback):
    """Test whether Treelite objects can be serialized to a byte sequence"""
    # pylint: disable=too-many-locals
    X, y = dataset
    n_classes = len(np.unique(y))
    kwargs = {"max_depth": max_depth, "random_state": 0}
    if clazz == HistGradientBoostingClassifier:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    else:
        kwargs["n_jobs"] = -1
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = callback.draw(
            sampled_from([None, DummyClassifier(strategy="prior"), "zero"])
        )
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)[:, np.newaxis, :]
    if (
        clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]
        and n_classes == 2
    ):
        expected_prob = expected_prob[:, :, 1:]

    # Prediction should be correct after a round-trip
    tl_model = treelite.sklearn.import_model(clf)

    serialized_bytes = tl_model.serialize_bytes()
    tl_model2 = treelite.Model.deserialize_bytes(serialized_bytes)
    out_prob = treelite.gtil.predict(tl_model2, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)

    # The model should serialize to the same byte sequence after a round-trip
    serialized_bytes2 = tl_model2.serialize_bytes()
    assert serialized_bytes == serialized_bytes2


@given(
    clazz=sampled_from(
        [
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
        ]
    ),
    n_estimators=integers(min_value=5, max_value=10),
    max_depth=integers(min_value=3, max_value=10),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_serialize_as_checkpoint(clazz, n_estimators, max_depth, callback):
    """Test whether Treelite objects can be serialized to a checkpoint"""
    # pylint: disable=too-many-locals
    if clazz in [RandomForestRegressor, ExtraTreesRegressor]:
        n_targets = callback.draw(integers(min_value=1, max_value=3))
    else:
        n_targets = callback.draw(just(1))
    X, y = callback.draw(standard_regression_datasets(n_targets=just(n_targets)))
    kwargs = {"max_depth": max_depth, "random_state": 0}
    if clazz == HistGradientBoostingRegressor:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingRegressor, HistGradientBoostingRegressor]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    else:
        kwargs["n_jobs"] = -1
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = callback.draw(
            sampled_from([None, DummyRegressor(strategy="mean"), "zero"])
        )
    clf = clazz(**kwargs)
    clf.fit(X, y)
    if n_targets > 1:
        expected_pred = clf.predict(X)[:, :, np.newaxis]
    else:
        expected_pred = clf.predict(X).reshape((X.shape[0], 1, -1))

    with TemporaryDirectory() as tmpdir:
        # Prediction should be correct after a round-trip
        tl_model = treelite.sklearn.import_model(clf)
        checkpoint_path = pathlib.Path(tmpdir) / "checkpoint.bin"
        tl_model.serialize(checkpoint_path)
        tl_model2 = treelite.Model.deserialize(checkpoint_path)
        out_pred = treelite.gtil.predict(tl_model2, X)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)

        # The model should serialize to the same byte sequence after a round-trip
        checkpoint_path2 = pathlib.Path(tmpdir) / "checkpoint2.bin"
        tl_model2.serialize(checkpoint_path2)
        with open(checkpoint_path, "rb") as f, open(checkpoint_path2, "rb") as f2:
            checkpoint = f.read()
            checkpoint2 = f2.read()
        assert checkpoint == checkpoint2
