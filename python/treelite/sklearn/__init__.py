# coding: utf-8
"""Converter to ingest scikit-learn models into Treelite"""

from __future__ import absolute_import as _abs

import ctypes
import numpy as np

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


def import_model_with_model_builder(sklearn_model):
    """
    Load a tree ensemble model from a scikit-learn model object using the model builder API.

    .. note:: Use ``import_model`` for production use

        This function exists to demonstrate the use of the model builder API and is slow with
        large models. For production, please use :py:func:`~treelite.sklearn.import_model`
        which is significantly faster.

    Parameters
    ----------
    sklearn_model : object of type \
                    :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                    :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesRegressor` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesClassifier` / \
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
      model = treelite.sklearn.import_model_with_model_builder(clf)
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


class ArrayOfArrays:
    """
    Utility class to marshall a collection of arrays in order to pass to a C function
    """
    def __init__(self, *, dtype):
        int64_ptr_type = ctypes.POINTER(ctypes.c_int64)
        float64_ptr_type = ctypes.POINTER(ctypes.c_double)
        if dtype == np.int64:
            self.ptr_type = int64_ptr_type
        elif dtype == np.float64:
            self.ptr_type = float64_ptr_type
        else:
            raise ValueError(f'dtype {dtype} is not supported')
        self.dtype = dtype
        self.collection = []

    def add(self, array, *, expected_shape=None):
        """Add an array to the collection"""
        assert array.dtype == self.dtype
        if expected_shape:
            assert array.shape == expected_shape, \
                    f'Expected shape: {expected_shape}, Got shape {array.shape}'
        v = np.array(array, copy=False, dtype=self.dtype, order='C')
        self.collection.append(v.ctypes.data_as(self.ptr_type))

    def as_c_array(self):
        """Prepare the collection to pass as an argument of a C function"""
        return c_array(self.ptr_type, self.collection)


def import_model(sklearn_model):
    # pylint: disable=R0914,R0912
    """
    Load a tree ensemble model from a scikit-learn model object

    Parameters
    ----------
    sklearn_model : object of type \
                    :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                    :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesRegressor` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesClassifier` / \
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
        raise TreeliteError('Not a scikit-learn model')

    if class_name in ['RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor',
                      'GradientBoostingClassifier']:
        leaf_value_expected_shape = lambda node_count: (node_count, 1, 1)
    elif class_name in ['RandomForestClassifier', 'ExtraTreesClassifier']:
        leaf_value_expected_shape = lambda node_count: (node_count, 1, sklearn_model.n_classes_)
    else:
        raise TreeliteError(f'Not supported: {class_name}')

    if class_name.startswith('GradientBoosting') and sklearn_model.init != 'zero':
        raise TreeliteError("Gradient boosted trees must be trained with the option init='zero'")

    node_count = []
    children_left = ArrayOfArrays(dtype=np.int64)
    children_right = ArrayOfArrays(dtype=np.int64)
    feature = ArrayOfArrays(dtype=np.int64)
    threshold = ArrayOfArrays(dtype=np.float64)
    value = ArrayOfArrays(dtype=np.float64)
    n_node_samples = ArrayOfArrays(dtype=np.int64)
    impurity = ArrayOfArrays(dtype=np.float64)
    for estimator in sklearn_model.estimators_:
        if class_name.startswith('GradientBoosting'):
            estimator_range = estimator
            learning_rate = sklearn_model.learning_rate
        else:
            estimator_range = [estimator]
            learning_rate = 1.0
        for sub_estimator in estimator_range:
            tree = sub_estimator.tree_
            node_count.append(tree.node_count)
            children_left.add(tree.children_left, expected_shape=(tree.node_count,))
            children_right.add(tree.children_right, expected_shape=(tree.node_count,))
            feature.add(tree.feature, expected_shape=(tree.node_count,))
            threshold.add(tree.threshold, expected_shape=(tree.node_count,))
            # Note: for gradient boosted trees, we shrink each leaf output by the learning rate
            value.add(tree.value * learning_rate,
                      expected_shape=leaf_value_expected_shape(tree.node_count))
            n_node_samples.add(tree.n_node_samples, expected_shape=(tree.node_count,))
            impurity.add(tree.impurity, expected_shape=(tree.node_count,))

    handle = ctypes.c_void_p()
    if class_name in ['RandomForestRegressor', 'ExtraTreesRegressor']:
        _check_call(_LIB.TreeliteLoadSKLearnRandomForestRegressor(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            c_array(ctypes.c_int64, node_count), children_left.as_c_array(),
            children_right.as_c_array(), feature.as_c_array(), threshold.as_c_array(),
            value.as_c_array(), n_node_samples.as_c_array(), impurity.as_c_array(),
            ctypes.byref(handle)))
    elif class_name in ['RandomForestClassifier', 'ExtraTreesClassifier']:
        _check_call(_LIB.TreeliteLoadSKLearnRandomForestClassifier(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            ctypes.c_int(sklearn_model.n_classes_), c_array(ctypes.c_int64, node_count),
            children_left.as_c_array(), children_right.as_c_array(), feature.as_c_array(),
            threshold.as_c_array(), value.as_c_array(), n_node_samples.as_c_array(),
            impurity.as_c_array(), ctypes.byref(handle)))
    elif class_name == 'GradientBoostingRegressor':
        _check_call(_LIB.TreeliteLoadSKLearnGradientBoostingRegressor(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            c_array(ctypes.c_int64, node_count), children_left.as_c_array(),
            children_right.as_c_array(), feature.as_c_array(), threshold.as_c_array(),
            value.as_c_array(), n_node_samples.as_c_array(), impurity.as_c_array(),
            ctypes.byref(handle)))
    elif class_name == 'GradientBoostingClassifier':
        _check_call(_LIB.TreeliteLoadSKLearnGradientBoostingClassifier(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            ctypes.c_int(sklearn_model.n_classes_), c_array(ctypes.c_int64, node_count),
            children_left.as_c_array(), children_right.as_c_array(), feature.as_c_array(),
            threshold.as_c_array(), value.as_c_array(), n_node_samples.as_c_array(),
            impurity.as_c_array(), ctypes.byref(handle)))
    return Model(handle)


__all__ = ['import_model_with_model_builder', 'import_model']
