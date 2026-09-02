"""Converter to export Treelite models as scikit-learn models (EXPERIMENTAL)"""

import warnings
from enum import IntEnum
from typing import Any

import numpy as np
from packaging.version import parse as parse_version

from ..core import TreeliteError
from ..model import Model


def _ensure_scalar_int(x: Any) -> int:
    if isinstance(x, np.ndarray):
        assert x.shape == (1,)
        return int(x[0])
    try:
        return int(x)
    except ValueError as e:
        raise ValueError(f"Cannot interpret x as a scalar integer, {x.type=}") from e


def _ensure_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    raise ValueError(f"x is not a valid NumPy array. {x.type=}")


# Fields of scikit-learn's private Node struct that this exporter knows how to
# handle. If scikit-learn grows a field outside this set we cannot populate it,
# and since we now allocate with scikit-learn's own NODE_DTYPE the dtype check in
# Tree.__setstate__ will no longer catch it -- the field would silently keep its
# zero value. `left_cat_bitset` is knowingly left zeroed: trees with categorical
# splits are rejected before we allocate.
_KNOWN_NODE_FIELDS = frozenset(
    {
        "left_child",
        "right_child",
        "feature",
        "threshold",
        "left_cat_bitset",
        "impurity",
        "n_node_samples",
        "weighted_n_node_samples",
        "missing_go_to_left",
        "split_kind",
    }
)


class _TaskType(IntEnum):
    # pylint: disable=invalid-name
    kBinaryClf = 0
    kRegressor = 1
    kMultiClf = 2
    kLearningToRank = 3
    kIsolationForest = 4


def _export_tree(
    model, *, tree_id, n_features, n_classes, n_targets, tree_depths, subestimator_class
):
    # pylint: disable=too-many-locals
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.tree._tree import NODE_DTYPE, TREE_LEAF
        from sklearn.tree._tree import Tree as SKLearnTree
    except ImportError as e:
        raise TreeliteError("This function requires scikit-learn package") from e

    tree_accessor = model.get_tree_accessor(tree_id)
    has_categorical_split = tree_accessor.get_field("has_categorical_split").tolist()[0]
    if has_categorical_split:
        raise NotImplementedError(
            "Trees with categorical splits cannot yet be exported as scikit-learn"
        )
    unknown_fields = sorted(set(NODE_DTYPE.names) - _KNOWN_NODE_FIELDS)
    if unknown_fields:
        raise NotImplementedError(
            f"scikit-learn {sklearn_version} stores tree node fields that this "
            f"version of Treelite does not know how to populate: {unknown_fields}. "
            "Exporting would produce a model that predicts incorrectly. "
            "Please upgrade Treelite."
        )

    n_nodes = tree_accessor.get_field("num_nodes").tolist()[0]
    if parse_version(sklearn_version) >= parse_version("1.10.0.dev0"):
        n_categories = np.full(n_features, -1, dtype=np.intp)
        tree = SKLearnTree(n_features, n_classes, n_targets, n_categories)
    else:
        tree = SKLearnTree(n_features, n_classes, n_targets)
    # Use scikit-learn's runtime node layout, since its private ABI can change
    # independently of the package version.
    nodes = np.zeros(n_nodes, dtype=NODE_DTYPE)

    nodes["left_child"] = tree_accessor.get_field("cleft")
    nodes["right_child"] = tree_accessor.get_field("cright")
    nodes["feature"] = tree_accessor.get_field("split_index")
    nodes["threshold"] = tree_accessor.get_field("threshold")
    nodes["impurity"] = np.nan
    data_count = tree_accessor.get_field("data_count").astype(np.intp)
    data_count_mask = tree_accessor.get_field("data_count_present").astype(np.bool_)
    if data_count.size == 0:
        nodes["n_node_samples"] = np.full((n_nodes,), fill_value=-1, dtype=np.intp)
    else:
        data_count[~data_count_mask] = -1
        nodes["n_node_samples"] = data_count
    # TODO(chyunsu3): In Treelite 5.0, rename field sum_hess -> weighted_data_count
    weighted_data_count = tree_accessor.get_field("sum_hess").astype(np.float64)
    weighted_data_count_mask = tree_accessor.get_field("sum_hess_present").astype(
        np.bool_
    )
    if weighted_data_count.size == 0:
        nodes["weighted_n_node_samples"] = np.full(
            (n_nodes,), fill_value=np.nan, dtype=np.float64
        )
    else:
        weighted_data_count[~weighted_data_count_mask] = np.nan
        nodes["weighted_n_node_samples"] = weighted_data_count
    nodes["missing_go_to_left"] = tree_accessor.get_field("default_left")
    if "split_kind" in nodes.dtype.names:
        from sklearn.tree._utils import SPLIT_LEAF, SPLIT_NUMERIC

        nodes["split_kind"] = SPLIT_NUMERIC
        nodes["split_kind"][nodes["left_child"] == TREE_LEAF] = SPLIT_LEAF

    if n_targets == 1 and n_classes[0] == 1:
        leaf_value = (
            tree_accessor.get_field("leaf_value").astype("float64").reshape((-1, 1, 1))
        )
    else:
        # Need to map leaf values to correct layout
        leaf_value = np.zeros((n_nodes, n_targets, n_classes[0]), dtype="float64")
        leaf_value_raw = tree_accessor.get_field("leaf_vector").astype("float64")
        leaf_vec_begin = tree_accessor.get_field("leaf_vector_begin")
        leaf_vec_end = tree_accessor.get_field("leaf_vector_end")
        for node_id in range(n_nodes):
            if leaf_vec_begin[node_id] != leaf_vec_end[node_id]:
                # This node is a leaf node and outputs a vector
                leaf_value[node_id, :, :] = leaf_value_raw[
                    leaf_vec_begin[node_id] : leaf_vec_end[node_id]
                ].reshape((n_targets, n_classes[0]))

    state = {
        "max_depth": tree_depths[tree_id],
        "node_count": n_nodes,
        "nodes": nodes,
        "values": leaf_value,
    }
    tree.__setstate__(state)

    subestimator = subestimator_class()
    subestimator_state = {
        "tree_": tree,
        "n_outputs_": n_targets,
        "is_categorical_": None,
        "_sklearn_version": sklearn_version,
    }
    if subestimator_class is DecisionTreeClassifier:
        if n_targets == 1:
            subestimator_state["n_classes_"] = n_classes[0]
        else:
            subestimator_state["n_classes_"] = n_classes.tolist()
    subestimator.__setstate__(subestimator_state)
    return subestimator


def export_model(model: Model) -> Any:
    """
    Export a model as a scikit-learn RandomForest.

    Note
    ----
    Currently only random forests and isolation forests can be exported as
    scikit-learn model objects.
    Support for gradient boosted trees and other kinds of tree models will be
    added in the future.

    Parameters
    ----------
    model : :py:class:`Model`
        Treelite mobel to export

    Returns
    -------
    sklearn_model : object of type \
                    :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                    :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                    :py:class:`~sklearn.ensemble.IsolationForest`
        Scikit-learn model
    """
    # pylint: disable=too-many-locals
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.ensemble import (
            IsolationForest,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.ensemble._iforest import _average_path_length
        from sklearn.tree import (
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            ExtraTreeRegressor,
        )
    except ImportError as e:
        raise TreeliteError("This function requires scikit-learn package") from e

    header_accessor = model.get_header_accessor()
    average_tree_output = (
        _ensure_scalar_int(header_accessor.get_field("average_tree_output")) == 1
    )
    task_type = _ensure_scalar_int(header_accessor.get_field("task_type"))
    n_features = _ensure_scalar_int(header_accessor.get_field("num_feature"))
    n_trees = _ensure_scalar_int(header_accessor.get_field("num_tree"))
    n_targets = _ensure_scalar_int(header_accessor.get_field("num_target"))
    n_classes = _ensure_numpy(header_accessor.get_field("num_class"))
    leaf_vector_shape = _ensure_numpy(header_accessor.get_field("leaf_vector_shape"))
    target_id = _ensure_numpy(header_accessor.get_field("target_id"))
    class_id = _ensure_numpy(header_accessor.get_field("class_id"))
    tree_depths = model.get_tree_depth()

    # Heuristics to ensure that the model can be represented as scikit-learn random forest
    # 1. average_tree_output must be True
    # 2. n_classes[i] must be identical for all targets
    # 3. Each leaf must yield an output of shape (n_targets, n_classes)
    # 4. target_id[i] must be either 0 or -1
    # 5. class_id[i] must be either 0 or -1
    def raise_not_rf_error(reason):
        raise NotImplementedError(
            "This Treelite model cannot be represented as scikit-learn random forest. "
            f"Condition unmet: {reason}"
            "Other kinds of tree models in scikit-learn are not yet supported."
        )

    if not average_tree_output:
        raise_not_rf_error(
            "Outputs of tree outputs must be averaged to produce the final output"
        )
    if not np.all(n_classes == n_classes[0]):
        raise_not_rf_error("n_classes must be identical for all trees")
    if not np.array_equal(leaf_vector_shape, [n_targets, n_classes.max()]):
        raise_not_rf_error(
            "Each tree must produce output of dimensions (n_targets, n_classes)"
        )
    if not np.all((target_id == 0) | (target_id == -1)):
        raise_not_rf_error("target_id field must be either 0 or -1")
    if not np.all((class_id == 0) | (class_id == -1)):
        raise_not_rf_error("class_id field must be either 0 or -1")

    if task_type in [_TaskType.kBinaryClf, _TaskType.kMultiClf]:
        estimator_class = RandomForestClassifier
        subestimator_class = DecisionTreeClassifier
    elif task_type == _TaskType.kIsolationForest:
        estimator_class = IsolationForest
        subestimator_class = ExtraTreeRegressor
    else:
        estimator_class = RandomForestRegressor
        subestimator_class = DecisionTreeRegressor

    estimators = []

    for tree_id in range(n_trees):
        estimators.append(
            _export_tree(
                model,
                tree_id=tree_id,
                n_features=n_features,
                n_classes=n_classes,
                n_targets=n_targets,
                tree_depths=tree_depths,
                subestimator_class=subestimator_class,
            )
        )

    clf = estimator_class()
    state = {
        "estimators_": estimators,
        "n_outputs_": n_targets,
        "n_features_in_": n_features,
        "_sklearn_version": sklearn_version,
    }
    if estimator_class is RandomForestClassifier:
        if n_targets == 1:
            state.update(
                {
                    "n_classes_": n_classes[0],
                    "classes_": np.arange(n_classes[0]),
                }
            )
        else:
            state.update(
                {
                    "n_classes_": n_classes.tolist(),
                    "classes_": [np.arange(n_classes[i]) for i in range(n_targets)],
                }
            )
    elif estimator_class is IsolationForest:
        # Recover the `offset_` field; if missing, set to -0.5
        try:
            offset = model.attributes["sklearn_iforest_offset"]
        except KeyError:
            warnings.warn(
                "Treelite model does not store attribute 'sklearn_iforest_offset'; "
                "setting it to the default value of -0.5...",
                UserWarning,
            )
            offset = -0.5

        # Compute max_samples by taking the max over the weighted root counts
        # (with bootstrap=True the unweighted root only counts distinct rows)
        max_samples = int(
            max(estimator.tree_.weighted_n_node_samples[0] for estimator in estimators)
        )
        state.update(
            {
                "_max_samples": max_samples,
                "max_samples_": max_samples,
                "offset_": offset,
                "_average_path_length_per_tree": tuple(
                    _average_path_length(est.tree_.n_node_samples) for est in estimators
                ),
                "_decision_path_lengths": tuple(
                    est.tree_.compute_node_depths() for est in estimators
                ),
                # The exported trees reference features globally, so scoring uses
                # the full feature set for every tree.
                "_max_features": n_features,
                "estimators_features_": [
                    np.arange(n_features, dtype=np.int64) for _ in estimators
                ],
            }
        )
    clf.__setstate__(state)

    return clf
