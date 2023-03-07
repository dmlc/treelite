# coding: utf-8
"""Converter to ingest scikit-learn models into Treelite"""

import ctypes

import numpy as np
from scipy.special import psi

from ..core import _LIB, _check_call, c_array
from ..frontend import Model
from ..util import TreeliteError


class ArrayOfArrays:
    """
    Utility class to marshall a collection of arrays in order to pass to a C function
    """

    def __init__(self, *, dtype):
        int8_ptr_type = ctypes.POINTER(ctypes.c_int8)
        int64_ptr_type = ctypes.POINTER(ctypes.c_int64)
        float64_ptr_type = ctypes.POINTER(ctypes.c_double)
        if dtype == np.int64:
            self.ptr_type = int64_ptr_type
        elif dtype == np.float64:
            self.ptr_type = float64_ptr_type
        elif dtype == np.int8:
            self.ptr_type = int8_ptr_type
        else:
            raise ValueError(f"dtype {dtype} is not supported")
        self.dtype = dtype
        self.collection = []
        self.collection_ptr = []

    def add(self, array, *, expected_shape=None):
        """Add an array to the collection"""
        assert array.dtype == self.dtype
        if expected_shape:
            assert (
                array.shape == expected_shape
            ), f"Expected shape: {expected_shape}, Got shape {array.shape}"
        v = np.array(array, copy=False, dtype=self.dtype, order="C")
        self.collection.append(v)

    def as_c_array(self):
        """Prepare the collection to pass as an argument of a C function"""
        for v in self.collection:
            self.collection_ptr.append(v.ctypes.data_as(self.ptr_type))
        return c_array(self.ptr_type, self.collection_ptr)


# Helpers for isolation forests
def harmonic(number):
    """Calculates the n-th harmonic number"""
    return psi(number + 1) + np.euler_gamma


def expected_depth(n_remainder):
    """Calculates the expected isolation depth for a remainder of uniform points"""
    if n_remainder <= 1:
        return 0
    if n_remainder == 2:
        return 1
    return float(2 * (harmonic(n_remainder) - 1))


def calculate_depths(isolation_depths, tree, curr_node, curr_depth):
    """Fill in an array of isolation depths for a scikit-learn isolation forest model"""
    if tree.children_left[curr_node] == -1:
        isolation_depths[curr_node] = curr_depth + expected_depth(
            tree.n_node_samples[curr_node]
        )
    else:
        calculate_depths(
            isolation_depths, tree, tree.children_left[curr_node], curr_depth + 1
        )
        calculate_depths(
            isolation_depths, tree, tree.children_right[curr_node], curr_depth + 1
        )


def import_model(sklearn_model):
    # pylint: disable=R0914,R0912,R0915
    """
    Load a tree ensemble model from a scikit-learn model object

    Note
    ----
    For 'IsolationForest', it will calculate the outlier score using the standardized ratio as
    proposed in the original reference, which matches with
    'IsolationForest._compute_chunked_score_samples' but is a bit different from
    'IsolationForest.decision_function'.

    Parameters
    ----------
    sklearn_model : object of type \
                    :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                    :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesRegressor` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesClassifier` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingRegressor` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingClassifier` / \
                    :py:class:`~sklearn.ensemble.HistGradientBoostingRegressor` / \
                    :py:class:`~sklearn.ensemble.HistGradientBoostingClassifier` / \
                    :py:class:`~sklearn.ensemble.IsolationForest`
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

    Notes
    -----
    This function does not yet support categorical splits in HistGradientBoostingRegressor and
    HistGradientBoostingClassifier. If you are using either estimator types, make sure that all
    test nodes have numerical test conditions.
    """
    try:
        from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesC
        from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesR
        from sklearn.ensemble import GradientBoostingClassifier as GradientBoostingC
        from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingR
        from sklearn.ensemble import (
            HistGradientBoostingClassifier as HistGradientBoostingC,
        )
        from sklearn.ensemble import (
            HistGradientBoostingRegressor as HistGradientBoostingR,
        )
        from sklearn.ensemble import IsolationForest
        from sklearn.ensemble import RandomForestClassifier as RandomForestC
        from sklearn.ensemble import RandomForestRegressor as RandomForestR
    except ImportError as e:
        raise TreeliteError("This function requires scikit-learn package") from e

    if isinstance(sklearn_model, (HistGradientBoostingR, HistGradientBoostingC)):
        return _import_hist_gradient_boosting(sklearn_model)

    if isinstance(
        sklearn_model,
        (
            RandomForestR,
            ExtraTreesR,
            GradientBoostingR,
            GradientBoostingC,
            IsolationForest,
        ),
    ):
        # pylint: disable=C3001
        leaf_value_expected_shape = lambda node_count: (node_count, 1, 1)
    elif isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        # pylint: disable=C3001
        leaf_value_expected_shape = lambda node_count: (
            node_count,
            1,
            sklearn_model.n_classes_,
        )
    else:
        raise TreeliteError(
            f"Not supported model type: {sklearn_model.__class__.__name__}"
        )

    if (
        isinstance(sklearn_model, (GradientBoostingR, GradientBoostingC))
        and sklearn_model.init != "zero"
    ):
        raise TreeliteError(
            "Gradient boosted trees must be trained with the option init='zero'"
        )

    if isinstance(sklearn_model, IsolationForest):
        ratio_c = expected_depth(sklearn_model.max_samples_)

    node_count = []
    children_left = ArrayOfArrays(dtype=np.int64)
    children_right = ArrayOfArrays(dtype=np.int64)
    feature = ArrayOfArrays(dtype=np.int64)
    threshold = ArrayOfArrays(dtype=np.float64)
    value = ArrayOfArrays(dtype=np.float64)
    n_node_samples = ArrayOfArrays(dtype=np.int64)
    weighted_n_node_samples = ArrayOfArrays(dtype=np.float64)
    impurity = ArrayOfArrays(dtype=np.float64)
    for estimator in sklearn_model.estimators_:
        if isinstance(sklearn_model, (GradientBoostingR, GradientBoostingC)):
            estimator_range = estimator
            learning_rate = sklearn_model.learning_rate
        else:
            estimator_range = [estimator]
            learning_rate = 1.0
        if isinstance(sklearn_model, IsolationForest):
            isolation_depths = np.zeros(
                estimator.tree_.n_node_samples.shape[0], dtype="float64"
            )
            calculate_depths(isolation_depths, estimator.tree_, 0, 0.0)
        for sub_estimator in estimator_range:
            tree = sub_estimator.tree_
            node_count.append(tree.node_count)
            children_left.add(tree.children_left, expected_shape=(tree.node_count,))
            children_right.add(tree.children_right, expected_shape=(tree.node_count,))
            feature.add(tree.feature, expected_shape=(tree.node_count,))
            threshold.add(tree.threshold, expected_shape=(tree.node_count,))
            if not isinstance(sklearn_model, IsolationForest):
                # Note: for gradient boosted trees, we shrink each leaf output by the
                # learning rate
                value.add(
                    tree.value * learning_rate,
                    expected_shape=leaf_value_expected_shape(tree.node_count),
                )
            else:
                value.add(
                    isolation_depths.reshape((-1, 1, 1)),
                    expected_shape=leaf_value_expected_shape(tree.node_count),
                )
            n_node_samples.add(tree.n_node_samples, expected_shape=(tree.node_count,))
            weighted_n_node_samples.add(
                tree.weighted_n_node_samples, expected_shape=(tree.node_count,)
            )
            impurity.add(tree.impurity, expected_shape=(tree.node_count,))

    handle = ctypes.c_void_p()
    if isinstance(sklearn_model, (RandomForestR, ExtraTreesR)):
        _check_call(
            _LIB.TreeliteLoadSKLearnRandomForestRegressor(
                ctypes.c_int(sklearn_model.n_estimators),
                ctypes.c_int(sklearn_model.n_features_in_),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                weighted_n_node_samples.as_c_array(),
                impurity.as_c_array(),
                ctypes.byref(handle),
            )
        )
    elif isinstance(sklearn_model, IsolationForest):
        _check_call(
            _LIB.TreeliteLoadSKLearnIsolationForest(
                ctypes.c_int(sklearn_model.n_estimators),
                ctypes.c_int(sklearn_model.n_features_in_),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                weighted_n_node_samples.as_c_array(),
                impurity.as_c_array(),
                ctypes.c_double(ratio_c),
                ctypes.byref(handle),
            )
        )
    elif isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        _check_call(
            _LIB.TreeliteLoadSKLearnRandomForestClassifier(
                ctypes.c_int(sklearn_model.n_estimators),
                ctypes.c_int(sklearn_model.n_features_in_),
                ctypes.c_int(sklearn_model.n_classes_),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                weighted_n_node_samples.as_c_array(),
                impurity.as_c_array(),
                ctypes.byref(handle),
            )
        )
    elif isinstance(sklearn_model, GradientBoostingR):
        _check_call(
            _LIB.TreeliteLoadSKLearnGradientBoostingRegressor(
                ctypes.c_int(sklearn_model.n_estimators),
                ctypes.c_int(sklearn_model.n_features_in_),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                weighted_n_node_samples.as_c_array(),
                impurity.as_c_array(),
                ctypes.byref(handle),
            )
        )
    elif isinstance(sklearn_model, GradientBoostingC):
        _check_call(
            _LIB.TreeliteLoadSKLearnGradientBoostingClassifier(
                ctypes.c_int(sklearn_model.n_estimators),
                ctypes.c_int(sklearn_model.n_features_in_),
                ctypes.c_int(sklearn_model.n_classes_),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                weighted_n_node_samples.as_c_array(),
                impurity.as_c_array(),
                ctypes.byref(handle),
            )
        )
    else:
        raise TreeliteError(
            f"Unsupported model type {sklearn_model.__class__.__name__}: "
            + "currently random forests, extremely randomized trees, and gradient "
            + "boosted trees are supported"
        )
    return Model(handle)


def _import_hist_gradient_boosting(sklearn_model):
    # pylint: disable=R0914,W0212
    """Load HistGradientBoostingClassifier / HistGradientBoostingRegressor"""
    from sklearn.ensemble import HistGradientBoostingClassifier as HistGradientBoostingC
    from sklearn.ensemble import HistGradientBoostingRegressor as HistGradientBoostingR

    # Arrays to be passed to C API functions
    node_count = []
    children_left = ArrayOfArrays(dtype=np.int64)
    children_right = ArrayOfArrays(dtype=np.int64)
    feature = ArrayOfArrays(dtype=np.int64)
    threshold = ArrayOfArrays(dtype=np.float64)
    default_left = ArrayOfArrays(dtype=np.int8)
    value = ArrayOfArrays(dtype=np.float64)
    n_node_samples = ArrayOfArrays(dtype=np.int64)
    gain = ArrayOfArrays(dtype=np.float64)

    for estimator in sklearn_model._predictors:
        estimator_range = estimator
        for sub_estimator in estimator_range:
            # each node has values:
            # ("value", "count", "feature_idx", "threshold", "missing_go_to_left",
            #  "left", "right", "gain", "depth", "is_leaf", "bin_threshold",
            #  "is_categorical", "bitset_idx")
            nodes = sub_estimator.nodes
            node_count.append(len(nodes))
            children_left.add(
                np.array([-1 if n[9] else n[5] for n in nodes], dtype=np.int64)
            )
            children_right.add(
                np.array([-1 if n[9] else n[6] for n in nodes], dtype=np.int64)
            )
            feature.add(np.array([-2 if n[9] else n[2] for n in nodes], dtype=np.int64))
            threshold.add(np.array([n[3] for n in nodes], dtype=np.float64))
            default_left.add(np.array([n[4] for n in nodes], dtype=np.int8))
            value.add(np.array([[n[0]] for n in nodes], dtype=np.float64))
            n_node_samples.add(np.array([[n[1]] for n in nodes], dtype=np.int64))
            gain.add(np.array([n[7] for n in nodes], dtype=np.float64))

            # TODO(hcho3): Add support for categorical splits
            for node in nodes:
                if node[11]:
                    raise NotImplementedError(
                        "Categorical splits are not yet supported for "
                        "HistGradientBoostingClassifier / HistGradientBoostingRegressor"
                    )

    handle = ctypes.c_void_p()
    if isinstance(sklearn_model, (HistGradientBoostingR,)):
        _check_call(
            _LIB.TreeliteLoadSKLearnHistGradientBoostingRegressor(
                ctypes.c_int(sklearn_model.n_iter_),
                ctypes.c_int(sklearn_model.n_features_in_),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                default_left.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                gain.as_c_array(),
                sklearn_model._baseline_prediction.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                ctypes.byref(handle),
            )
        )
    elif isinstance(sklearn_model, (HistGradientBoostingC,)):
        _check_call(
            _LIB.TreeliteLoadSKLearnHistGradientBoostingClassifier(
                ctypes.c_int(sklearn_model.n_iter_),
                ctypes.c_int(sklearn_model.n_features_in_),
                ctypes.c_int(len(sklearn_model.classes_)),
                c_array(ctypes.c_int64, node_count),
                children_left.as_c_array(),
                children_right.as_c_array(),
                feature.as_c_array(),
                threshold.as_c_array(),
                default_left.as_c_array(),
                value.as_c_array(),
                n_node_samples.as_c_array(),
                gain.as_c_array(),
                sklearn_model._baseline_prediction.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_double)
                ),
                ctypes.byref(handle),
            )
        )
    else:
        raise TreeliteError(
            f"Unsupported model type {sklearn_model.__class__.__name__}"
        )
    return Model(handle)
