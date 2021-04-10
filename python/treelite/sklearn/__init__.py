# coding: utf-8
"""Converter to ingest scikit-learn models into Treelite"""

from __future__ import absolute_import as _abs

from ..util import TreeliteError
from ..core import _LIB, c_array, _check_call
from ..frontend import Model
from .common import SKLConverterBase
from .gbm_regressor import SKLGBMRegressorMixin
from .gbm_classifier import SKLGBMClassifierMixin
from .gbm_multi_classifier import SKLGBMMultiClassifierMixin
from .rf_regressor import SKLRFRegressorMixin
from .rf_classifier import SKLRFClassifierMixin
from .rf_multi_classifier import SKLRFMultiClassifierMixin
import ctypes
import numpy as np


def import_model(sklearn_model):
    """
    Load a tree ensemble model from a scikit-learn model object

    Parameters
    ----------
    sklearn_model : object of type \
                    :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                    :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingRegressor` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingClassifier`
        Python handle to scikit-learn model

    Returns
    -------
    model : :py:class:`~treelite.Model` object
        loaded model

    Example
    -------

    .. code-block:: python
      :emphasize-lines: 8

      import sklearn.datasets
      import sklearn.ensemble
      X, y = sklearn.datasets.load_boston(return_X_y=True)
      clf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
      clf.fit(X, y)

      import treelite.sklearn
      model = treelite.sklearn.import_model(clf)
    """
    class_name = sklearn_model.__class__.__name__
    module_name = sklearn_model.__module__.split('.')[0]

    if module_name != 'sklearn':
        raise Exception('Not a scikit-learn model')

    if class_name in ['RandomForestRegressor', 'ExtraTreesRegressor']:
        return SKLRFRegressorConverter.process_model(sklearn_model)
    if class_name in ['RandomForestClassifier', 'ExtraTreesClassifier']:
        if sklearn_model.n_classes_ == 2:
            return SKLRFClassifierConverter.process_model(sklearn_model)
        if sklearn_model.n_classes_ > 2:
            return SKLRFMultiClassifierConverter.process_model(sklearn_model)
        raise TreeliteError('n_classes_ must be at least 2')
    if class_name == 'GradientBoostingRegressor':
        return SKLGBMRegressorConverter.process_model(sklearn_model)
    if class_name == 'GradientBoostingClassifier':
        if sklearn_model.n_classes_ == 2:
            return SKLGBMClassifierConverter.process_model(sklearn_model)
        if sklearn_model.n_classes_ > 2:
            return SKLGBMMultiClassifierConverter.process_model(sklearn_model)
        raise TreeliteError('n_classes_ must be at least 2')
    raise TreeliteError('Unsupported model type: currently ' +
                        'random forests, extremely randomized trees, and gradient boosted trees ' +
                        'are supported')


class SKLGBMRegressorConverter(SKLGBMRegressorMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLGBMClassifierConverter(SKLGBMClassifierMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLGBMMultiClassifierConverter(SKLGBMMultiClassifierMixin, SKLConverterBase):
    # pylint: disable=C0111
    pass


class SKLRFRegressorConverter(SKLRFRegressorMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLRFClassifierConverter(SKLRFClassifierMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLRFMultiClassifierConverter(SKLRFMultiClassifierMixin, SKLConverterBase):
    # pylint: disable=C0111
    pass


def import_model_v2(clf):
    int64_ptr_type = ctypes.POINTER(ctypes.c_int64)
    float64_ptr_type = ctypes.POINTER(ctypes.c_double)

    node_count = []
    children_left = []
    children_right = []
    feature = []
    threshold = []
    value = []
    n_node_samples = []
    impurity = []
    for i, estimator in enumerate(clf.estimators_):
        tree = estimator.tree_
        node_count_v = tree.node_count
        node_count.append(node_count_v)
        assert tree.children_left.shape == (node_count_v,)
        children_left_v = np.array(tree.children_left, copy=False, dtype=np.int64, order='C')
        children_left.append(children_left_v.ctypes.data_as(int64_ptr_type))
        assert tree.children_right.shape == (node_count_v,)
        children_right_v = np.array(tree.children_right, copy=False, dtype=np.int64, order='C')
        children_right.append(children_right_v.ctypes.data_as(int64_ptr_type))
        assert tree.feature.shape == (node_count_v,)
        feature_v = np.array(tree.feature, copy=False, dtype=np.int64, order='C')
        feature.append(feature_v.ctypes.data_as(int64_ptr_type))
        assert tree.threshold.shape == (node_count_v,)
        threshold_v = np.array(tree.threshold, copy=False, dtype=np.float64, order='C')
        threshold.append(threshold_v.ctypes.data_as(float64_ptr_type))
        assert tree.value.shape == (node_count_v, 1, 1)
        value_v = np.array(tree.value, copy=False, dtype=np.float64, order='C')
        value.append(value_v.ctypes.data_as(float64_ptr_type))
        assert tree.n_node_samples.shape == (node_count_v,)
        n_node_samples_v = np.array(tree.n_node_samples, copy=False, dtype=np.int64, order='C')
        n_node_samples.append(n_node_samples_v.ctypes.data_as(int64_ptr_type))
        assert tree.impurity.shape == (node_count_v,)
        impurity_v = np.array(tree.impurity, copy=False, dtype=np.float64, order='C')
        impurity.append(impurity_v.ctypes.data_as(float64_ptr_type))

    handle = ctypes.c_void_p()
    _check_call(_LIB.TreeliteLoadSKLearnRandomForestRegressor(
        ctypes.c_int(clf.n_estimators), ctypes.c_int(clf.n_features_),
        c_array(ctypes.c_int64, node_count), c_array(int64_ptr_type, children_left),
        c_array(int64_ptr_type, children_right), c_array(int64_ptr_type, feature),
        c_array(float64_ptr_type, threshold), c_array(float64_ptr_type, value),
        c_array(int64_ptr_type, n_node_samples), c_array(float64_ptr_type, impurity),
        ctypes.byref(handle)))
    return Model(handle)


__all__ = ['import_model', 'import_model_v2']
