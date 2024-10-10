"""Converter to export Treelite models as scikit-learn models (EXPERIMENTAL)"""

from enum import IntEnum
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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


_node_dtype = np.dtype(
    {
        "names": [
            "left_child",
            "right_child",
            "feature",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "missing_go_to_left",
        ],
        "formats": ["<i8", "<i8", "<i8", "<f8", "<f8", "<i8", "<f8", "u1"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56],
        "itemsize": 64,
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
        from sklearn.tree._tree import Tree as SKLearnTree
    except ImportError as e:
        raise TreeliteError("This function requires scikit-learn package") from e

    tree_accessor = model.get_tree_accessor(tree_id)
    has_categorical_split = tree_accessor.get_field("has_categorical_split").tolist()[0]
    if has_categorical_split:
        raise NotImplementedError(
            "Trees with categorical splits cannot yet be exported as scikit-learn"
        )

    tree = SKLearnTree(n_features, n_classes, n_targets)

    n_nodes = tree_accessor.get_field("num_nodes").tolist()[0]
    nodes = np.empty(n_nodes, dtype=_node_dtype)

    nodes["left_child"] = tree_accessor.get_field("cleft")
    nodes["right_child"] = tree_accessor.get_field("cright")
    nodes["feature"] = tree_accessor.get_field("split_index")
    nodes["threshold"] = tree_accessor.get_field("threshold")
    nodes["impurity"] = np.nan
    nodes["n_node_samples"] = -1
    nodes["weighted_n_node_samples"] = np.nan
    nodes["missing_go_to_left"] = tree_accessor.get_field("default_left")

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
        "_sklearn_version": sklearn_version,
    }
    if subestimator_class is DecisionTreeClassifier:
        if n_targets == 1:
            subestimator_state["n_classes_"] = n_classes[0]
        else:
            subestimator_state["n_classes_"] = n_classes.tolist()
    subestimator.__setstate__(subestimator_state)
    return subestimator


def export_model(model: Model):
    """
    Export a model as a scikit-learn RandomForest.

    Note
    ----
    Currently only random forests can be exported as scikit-learn model objects.
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
                    :py:class:`~sklearn.ensemble.GradientBoostingRegressor` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingClassifier`
        Scikit-learn model
    """
    # pylint: disable=too-many-locals
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.ensemble import RandomForestRegressor
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
    clf.__setstate__(state)

    return clf
