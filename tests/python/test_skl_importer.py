# -*- coding: utf-8 -*-
"""Tests for scikit-learn importer"""
import os

import pytest
import numpy as np
import treelite
import treelite_runtime
from treelite.contrib import _libext
from treelite.util import has_sklearn
from .util import os_compatible_toolchains


if has_sklearn():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
        ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, \
        ExtraTreesRegressor
    from sklearn.datasets import load_iris, load_breast_cancer, load_boston
else:
    class RandomForestClassifier:  # pylint: disable=missing-class-docstring, R0903
        pass


    class GradientBoostingClassifier:  # pylint: disable=missing-class-docstring, R0903
        pass


    class ExtraTreesClassifier:  # pylint: disable=missing-class-docstring, R0903
        pass


    class ExtraTreesRegressor:  # pylint: disable=missing-class-docstring, R0903
        pass


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('clazz', [RandomForestClassifier, ExtraTreesClassifier,
                                   GradientBoostingClassifier])
def test_skl_converter_multiclass_classifier(tmpdir, clazz, toolchain):
    # pylint: disable=too-many-locals
    """Convert scikit-learn multi-class classifier"""
    X, y = load_iris(return_X_y=True)
    kwargs = {}
    if clazz == GradientBoostingClassifier:
        kwargs['init'] = 'zero'
    clf = clazz(max_depth=3, random_state=0, n_estimators=10, **kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

    if clazz == RandomForestClassifier:
        model = treelite.sklearn.import_model_v2(clf)
    else:
        model = treelite.sklearn.import_model(clf)
    assert model.num_feature == clf.n_features_
    assert model.num_class == clf.n_classes_
    assert (model.num_tree ==
            clf.n_estimators * (clf.n_classes_ if clazz == GradientBoostingClassifier else 1))

    dtrain = treelite_runtime.DMatrix(X, dtype='float64')
    annotation_path = os.path.join(tmpdir, 'annotation.json')
    annotator = treelite.Annotator()
    annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
    annotator.save(path=annotation_path)

    libpath = os.path.join(tmpdir, 'skl' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={'annotate_in': annotation_path},
                     verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath)
    assert predictor.num_feature == clf.n_features_
    assert predictor.num_class == clf.n_classes_
    assert (predictor.pred_transform ==
            ('softmax' if clazz == GradientBoostingClassifier else 'identity_multiclass'))
    assert predictor.global_bias == 0.0
    assert predictor.sigmoid_alpha == 1.0
    out_prob = predictor.predict(dtrain)
    np.testing.assert_almost_equal(out_prob, expected_prob)


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('clazz', [RandomForestClassifier, ExtraTreesClassifier,
                                   GradientBoostingClassifier])
def test_skl_converter_binary_classifier(tmpdir, clazz, toolchain):
    # pylint: disable=too-many-locals
    """Convert scikit-learn binary classifier"""
    X, y = load_breast_cancer(return_X_y=True)
    kwargs = {}
    if clazz == GradientBoostingClassifier:
        kwargs['init'] = 'zero'
    clf = clazz(max_depth=3, random_state=0, n_estimators=10, **kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)[:, 1]

    if clazz == RandomForestClassifier:
        model = treelite.sklearn.import_model_v2(clf)
    else:
        model = treelite.sklearn.import_model(clf)
    assert model.num_feature == clf.n_features_
    assert model.num_class == 1
    assert model.num_tree == clf.n_estimators

    dtrain = treelite_runtime.DMatrix(X, dtype='float64')
    annotation_path = os.path.join(tmpdir, 'annotation.json')
    annotator = treelite.Annotator()
    annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
    annotator.save(path=annotation_path)

    libpath = os.path.join(tmpdir, 'skl' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={'annotate_in': annotation_path},
                     verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath)
    assert predictor.num_feature == clf.n_features_
    assert model.num_class == 1
    assert (predictor.pred_transform
            == ('sigmoid' if clazz == GradientBoostingClassifier else 'identity'))
    assert predictor.global_bias == 0.0
    assert predictor.sigmoid_alpha == 1.0
    out_prob = predictor.predict(dtrain)
    np.testing.assert_almost_equal(out_prob, expected_prob)


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('clazz', [RandomForestRegressor, ExtraTreesRegressor,
                                   GradientBoostingRegressor])
def test_skl_converter_regressor(tmpdir, clazz, toolchain):  # pylint: disable=too-many-locals
    """Convert scikit-learn regressor"""
    X, y = load_boston(return_X_y=True)
    kwargs = {}
    if clazz == GradientBoostingRegressor:
        kwargs['init'] = 'zero'
    clf = clazz(max_depth=3, random_state=0, n_estimators=10, **kwargs)
    clf.fit(X, y)
    expected_pred = clf.predict(X)

    if clazz == RandomForestRegressor:
        model = treelite.sklearn.import_model_v2(clf)
    else:
        model = treelite.sklearn.import_model(clf)
    assert model.num_feature == clf.n_features_
    assert model.num_class == 1
    assert model.num_tree == clf.n_estimators

    dtrain = treelite_runtime.DMatrix(X, dtype='float32')
    annotation_path = os.path.join(tmpdir, 'annotation.json')
    annotator = treelite.Annotator()
    annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
    annotator.save(path=annotation_path)

    libpath = os.path.join(tmpdir, 'skl' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={'annotate_in': annotation_path},
                     verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath)
    assert predictor.num_feature == clf.n_features_
    assert model.num_class == 1
    assert predictor.pred_transform == 'identity'
    assert predictor.global_bias == 0.0
    assert predictor.sigmoid_alpha == 1.0
    out_pred = predictor.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)
