# coding: utf-8
"""Converter to ingest scikit-learn models into Treelite"""

import ctypes
import numpy as np

from ..util import TreeliteError
from ..core import _LIB, c_array, _check_call
from ..frontend import Model


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
    try:
        import sklearn.ensemble
        from sklearn.ensemble import RandomForestRegressor as RandomForestR
        from sklearn.ensemble import RandomForestClassifier as RandomForestC
        from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesR
        from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesC
        from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingR
        from sklearn.ensemble import GradientBoostingClassifier as GradientBoostingC
    except ImportError as e:
        raise TreeliteError('This function requires scikit-learn package') from e

    if isinstance(sklearn_model,
            (RandomForestR, ExtraTreesR, GradientBoostingR, GradientBoostingC)):
        leaf_value_expected_shape = lambda node_count: (node_count, 1, 1)
    elif isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        leaf_value_expected_shape = lambda node_count: (node_count, 1, sklearn_model.n_classes_)
    else:
        raise TreeliteError(f'Not supported model type: {sklearn_model.__class__.__name__}')

    if isinstance(sklearn_model,
            (GradientBoostingR, GradientBoostingC)) and sklearn_model.init != 'zero':
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
        if isinstance(sklearn_model, (GradientBoostingR, GradientBoostingC)):
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
    if isinstance(sklearn_model, (RandomForestR, ExtraTreesR)):
        _check_call(_LIB.TreeliteLoadSKLearnRandomForestRegressor(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            c_array(ctypes.c_int64, node_count), children_left.as_c_array(),
            children_right.as_c_array(), feature.as_c_array(), threshold.as_c_array(),
            value.as_c_array(), n_node_samples.as_c_array(), impurity.as_c_array(),
            ctypes.byref(handle)))
    elif isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        _check_call(_LIB.TreeliteLoadSKLearnRandomForestClassifier(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            ctypes.c_int(sklearn_model.n_classes_), c_array(ctypes.c_int64, node_count),
            children_left.as_c_array(), children_right.as_c_array(), feature.as_c_array(),
            threshold.as_c_array(), value.as_c_array(), n_node_samples.as_c_array(),
            impurity.as_c_array(), ctypes.byref(handle)))
    elif isinstance(sklearn_model, GradientBoostingR):
        _check_call(_LIB.TreeliteLoadSKLearnGradientBoostingRegressor(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            c_array(ctypes.c_int64, node_count), children_left.as_c_array(),
            children_right.as_c_array(), feature.as_c_array(), threshold.as_c_array(),
            value.as_c_array(), n_node_samples.as_c_array(), impurity.as_c_array(),
            ctypes.byref(handle)))
    elif isinstance(sklearn_model, GradientBoostingC):
        _check_call(_LIB.TreeliteLoadSKLearnGradientBoostingClassifier(
            ctypes.c_int(sklearn_model.n_estimators), ctypes.c_int(sklearn_model.n_features_),
            ctypes.c_int(sklearn_model.n_classes_), c_array(ctypes.c_int64, node_count),
            children_left.as_c_array(), children_right.as_c_array(), feature.as_c_array(),
            threshold.as_c_array(), value.as_c_array(), n_node_samples.as_c_array(),
            impurity.as_c_array(), ctypes.byref(handle)))
    else:
        raise TreeliteError(f'Unsupported model type {sklearn_model.__class__.__name__}: ' +
                            'currently random forests, extremely randomized trees, and gradient ' +
                            'boosted trees are supported')
    return Model(handle)
