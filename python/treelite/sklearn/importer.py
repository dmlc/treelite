"""Converter to ingest scikit-learn models into Treelite"""

import ctypes

import numpy as np
from packaging.version import parse as parse_version

from ..core import _LIB, TreeliteError, _check_call
from ..frontend import Model
from ..util import c_array
from .isolation_forest import calculate_depths, expected_depth


class ArrayOfArrays:
    """
    Utility class to marshall a collection of arrays in order to pass to a C function
    """

    def __init__(self, *, dtype):
        int8_ptr_type = ctypes.POINTER(ctypes.c_int8)
        int64_ptr_type = ctypes.POINTER(ctypes.c_int64)
        uint32_ptr_type = ctypes.POINTER(ctypes.c_uint32)
        float64_ptr_type = ctypes.POINTER(ctypes.c_double)
        void_ptr_type = ctypes.c_void_p
        if dtype == np.int64:
            self.ptr_type = int64_ptr_type
        elif dtype == np.float64:
            self.ptr_type = float64_ptr_type
        elif dtype == np.uint32:
            self.ptr_type = uint32_ptr_type
        elif dtype == np.int8:
            self.ptr_type = int8_ptr_type
        elif dtype == "void":
            self.ptr_type = void_ptr_type
        else:
            raise ValueError(f"dtype {dtype} is not supported")
        self.dtype = dtype
        self.collection = []
        self.collection_ptr = []

    def add(self, array, *, expected_shape=None):
        """Add an array to the collection"""
        if self.dtype != "void":
            assert array.dtype == self.dtype
        if expected_shape:
            assert (
                array.shape == expected_shape
            ), f"Expected shape: {expected_shape}, Got shape {array.shape}"
        v = np.asarray(array, dtype=self.dtype, order="C")
        self.collection.append(v)

    def as_c_array(self):
        """Prepare the collection to pass as an argument of a C function"""
        if not self.collection:
            return None  # return nullptr to represent empty array
        for v in self.collection:
            self.collection_ptr.append(v.ctypes.data_as(self.ptr_type))
        return c_array(self.ptr_type, self.collection_ptr)


def import_model(sklearn_model):
    # pylint: disable=R0914,R0912,R0915
    """
    Load a tree ensemble model from a scikit-learn model object

    Note
    ----
    For :py:class:`~sklearn.ensemble.IsolationForest`, the loaded model will calculate the outlier
    score using the standardized ratio as proposed in the original reference,
    which matches with :py:meth:`~sklearn.ensemble.IsolationForest._compute_chunked_score_samples`
    but is a bit different from :py:meth:`~sklearn.ensemble.IsolationForest.decision_function`.

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
    model : :py:class:`Model`
        Loaded model

    Example
    -------

    .. code-block:: python
      :emphasize-lines: 8

      import sklearn.datasets
      import sklearn.ensemble
      X, y = sklearn.datasets.load_diabetes(return_X_y=True)
      clf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
      clf.fit(X, y)

      import treelite.sklearn
      model = treelite.sklearn.import_model(clf)

    Note
    ----
    This function does not yet support categorical splits in
    :py:class:`~sklearn.ensemble.HistGradientBoostingRegressor` and
    :py:class:`~sklearn.ensemble.HistGradientBoostingClassifier`.
    If you are using either estimator types, make sure that all
    test nodes have numerical test conditions.
    """
    try:
        from sklearn.dummy import DummyClassifier, DummyRegressor
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

    if isinstance(sklearn_model, (RandomForestR, ExtraTreesR)):
        # pylint: disable=C3001
        leaf_value_expected_shape = lambda node_count: (  # noqa: E731
            node_count,
            sklearn_model.n_outputs_,
            1,
        )
    elif isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        # pylint: disable=C3001
        try:  # multi-target
            max_n_classes = max(sklearn_model.n_classes_)
        except TypeError:  # single-target
            max_n_classes = sklearn_model.n_classes_
        leaf_value_expected_shape = lambda node_count: (  # noqa: E731
            node_count,
            sklearn_model.n_outputs_,
            max_n_classes,
        )
    elif isinstance(
        sklearn_model, (GradientBoostingR, GradientBoostingC, IsolationForest)
    ):
        # pylint: disable=C3001
        leaf_value_expected_shape = lambda node_count: (  # noqa: E731
            node_count,
            1,
            1,
        )
    else:
        raise TreeliteError(
            f"Not supported model type: {sklearn_model.__class__.__name__}"
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
            if isinstance(sklearn_model, IsolationForest):
                value.add(
                    isolation_depths.reshape((-1, 1, 1)),
                    expected_shape=leaf_value_expected_shape(tree.node_count),
                )
            else:
                # Note: for gradient boosted trees, we shrink each leaf output by the
                # learning rate
                value.add(
                    tree.value * learning_rate,
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
                ctypes.c_int(sklearn_model.n_outputs_),
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
        n_classes = np.array(sklearn_model.n_classes_, dtype=np.int32)
        _check_call(
            _LIB.TreeliteLoadSKLearnRandomForestClassifier(
                ctypes.c_int(sklearn_model.n_estimators),
                ctypes.c_int(sklearn_model.n_features_in_),
                ctypes.c_int(sklearn_model.n_outputs_),
                n_classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
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
        if sklearn_model.init_ == "zero":
            base_scores = np.array([0], dtype=np.float64)
        elif isinstance(sklearn_model.init_, (DummyRegressor,)):
            base_scores = np.array(sklearn_model.init_.constant_, dtype=np.float64)
            assert base_scores.size == 1
        else:
            raise NotImplementedError("Custom init estimator not supported")
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
                base_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(handle),
            )
        )
    elif isinstance(sklearn_model, GradientBoostingC):
        if sklearn_model.init_ == "zero":
            base_scores = np.zeros(shape=(sklearn_model.n_classes_,), dtype=np.float64)
        elif isinstance(sklearn_model.init_, (DummyClassifier,)):
            if sklearn_model.init_.strategy != "prior":
                raise NotImplementedError("Custom init estimator not supported")
            # pylint: disable=W0212
            base_scores = np.array(
                sklearn_model._raw_predict_init(
                    np.zeros((1, sklearn_model.n_features_in_))
                ),
                dtype=np.float64,
            )
        else:
            raise NotImplementedError("Custom init estimator not supported")
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
                base_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(handle),
            )
        )
    else:
        raise TreeliteError(
            f"Unsupported model type {sklearn_model.__class__.__name__}: "
            + "currently random forests, extremely randomized trees, and gradient "
            + "boosted trees are supported"
        )
    return Model(handle=handle)


def _import_hist_gradient_boosting(sklearn_model):
    # pylint: disable=R0914,W0212,too-many-branches
    """Load HistGradientBoostingClassifier / HistGradientBoostingRegressor"""
    from sklearn.ensemble import HistGradientBoostingClassifier as HistGradientBoostingC
    from sklearn.ensemble import HistGradientBoostingRegressor as HistGradientBoostingR

    # Arrays to be passed to C API functions
    (
        known_cat_bitsets,
        f_idx_map,
    ) = sklearn_model._bin_mapper.make_known_categories_bitsets()

    cat_remapper = ArrayOfArrays(dtype=np.int64)

    from sklearn import __version__ as sklearn_version

    if (
        parse_version(sklearn_version) >= parse_version("1.4.0")
        and getattr(sklearn_model, "_preprocessor", None) is not None
    ):
        # Ensure that the model's internals did not change
        assert len(sklearn_model._preprocessor.transformers_) == 2
        assert len(sklearn_model._preprocessor.transformers_[0]) == 3
        assert len(sklearn_model._preprocessor.transformers_[1]) == 3
        assert sklearn_model._preprocessor.transformers_[0][0] == "encoder"
        assert sklearn_model._preprocessor.transformers_[1][0] == "numerical"

        for cats in sklearn_model._preprocessor.transformers_[0][1].categories_:
            if cats.dtype.type is np.str_:
                raise NotImplementedError("String categories are not supported")
            cats = cats.astype(np.int64)
            cat_remapper.add(cats)

        feat_remapper = np.zeros((sklearn_model.n_features_in_,), dtype=np.int32)
        n_categorical = sklearn_model.is_categorical_.sum()
        cat_idx, num_idx = 0, 0
        for i, e in enumerate(sklearn_model.is_categorical_):
            if e:
                feat_remapper[cat_idx] = i
                cat_idx += 1
            else:
                feat_remapper[n_categorical + num_idx] = i
                num_idx += 1
    else:
        feat_remapper = np.arange(
            start=0, stop=sklearn_model.n_features_in_, dtype=np.int32
        )

    n_categorical_splits = known_cat_bitsets.shape[0]
    n_trees = 0
    nodes = ArrayOfArrays(dtype="void")
    raw_left_cat_bitsets = ArrayOfArrays(dtype=np.uint32)
    node_count = []
    itemsize = None

    for estimator in sklearn_model._predictors:
        estimator_range = estimator
        for sub_estimator in estimator_range:
            nodes.add(sub_estimator.nodes)
            raw_left_cat_bitsets.add(sub_estimator.raw_left_cat_bitsets)
            node_count.append(len(sub_estimator.nodes))
            n_trees += 1
            if itemsize is None:
                itemsize = sub_estimator.nodes.itemsize
            elif itemsize != sub_estimator.nodes.itemsize:
                raise RuntimeError("itemsize mismatch")

    handle = ctypes.c_void_p()
    if isinstance(sklearn_model, (HistGradientBoostingR,)):
        _check_call(
            _LIB.TreeliteLoadSKLearnHistGradientBoostingRegressor(
                ctypes.c_int(sklearn_model.n_iter_),
                ctypes.c_int(sklearn_model.n_features_in_),
                c_array(ctypes.c_int64, node_count),
                nodes.as_c_array(),
                ctypes.c_int(itemsize),
                ctypes.c_int32(n_categorical_splits),
                raw_left_cat_bitsets.as_c_array(),
                known_cat_bitsets.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                f_idx_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                feat_remapper.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                cat_remapper.as_c_array(),
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
                nodes.as_c_array(),
                ctypes.c_int(itemsize),
                ctypes.c_int32(n_categorical_splits),
                raw_left_cat_bitsets.as_c_array(),
                known_cat_bitsets.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                f_idx_map.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                feat_remapper.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                cat_remapper.as_c_array(),
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
    return Model(handle=handle)
