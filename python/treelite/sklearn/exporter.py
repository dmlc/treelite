"""Converter to export Treelite models as scikit-learn models (EXPERIMENTAL)"""

from typing import Any

import numpy as np

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


def export_model(model: Model):
    """
    Export as scikit-learn RandomForest or GradientBoosting

    Parameters
    ----------
    model :py:class:`Model`
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
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.tree._tree import Tree as SKLearnTree
    except ImportError as e:
        raise TreeliteError("This function requires scikit-learn package") from e

    node_dtype = np.dtype(
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

    header_accessor = model.get_header_accessor()
    # average_tree_output = (
    #    header_accessor.get_field("average_tree_output").tolist()[0] == 1
    # )
    n_features = _ensure_scalar_int(header_accessor.get_field("num_feature"))
    n_trees = _ensure_scalar_int(header_accessor.get_field("num_tree"))
    n_targets = _ensure_scalar_int(header_accessor.get_field("num_target"))
    n_classes = _ensure_numpy(header_accessor.get_field("num_class"))
    leaf_vector_shape = _ensure_numpy(header_accessor.get_field("leaf_vector_shape"))
    tree_depths = model.get_tree_depth()

    assert np.all(n_classes == 1)
    assert np.array_equal(leaf_vector_shape, [n_targets, 1])

    estimators = []

    for tree_id in range(n_trees):
        tree_accessor = model.get_tree_accessor(tree_id)
        has_categorical_split = tree_accessor.get_field(
            "has_categorical_split"
        ).tolist()[0]
        assert not has_categorical_split

        tree = SKLearnTree(n_features, n_classes, n_targets)

        n_nodes = tree_accessor.get_field("num_nodes").tolist()[0]
        nodes = np.empty(n_nodes, dtype=node_dtype)

        nodes["left_child"] = tree_accessor.get_field("cleft")
        nodes["right_child"] = tree_accessor.get_field("cright")
        nodes["feature"] = tree_accessor.get_field("split_index")
        nodes["threshold"] = tree_accessor.get_field("threshold")
        nodes["impurity"] = np.nan
        nodes["n_node_samples"] = -1
        nodes["weighted_n_node_samples"] = np.nan
        nodes["missing_go_to_left"] = tree_accessor.get_field("default_left")

        if n_targets == 1:
            leaf_value = (
                tree_accessor.get_field("leaf_value")
                .astype("float64")
                .reshape((-1, 1, 1))
            )
        else:
            # Need to map leaf values to correct layout
            leaf_value = np.zeros((n_nodes, n_targets, 1), dtype="float64")
            leaf_value_raw = tree_accessor.get_field("leaf_vector").astype("float64")
            leaf_vec_begin = tree_accessor.get_field("leaf_vector_begin")
            leaf_vec_end = tree_accessor.get_field("leaf_vector_end")
            for node_id in range(n_nodes):
                if leaf_vec_begin[node_id] != leaf_vec_end[node_id]:
                    # This node is a leaf node and outputs a vector
                    leaf_value[node_id, :, :] = leaf_value_raw[
                        leaf_vec_begin[node_id] : leaf_vec_end[node_id]
                    ].reshape((n_targets, 1))

        state = {
            "max_depth": tree_depths[tree_id],
            "node_count": n_nodes,
            "nodes": nodes,
            "values": leaf_value,
        }
        tree.__setstate__(state)

        reg = DecisionTreeRegressor()
        reg.__setstate__(
            {
                "tree_": tree,
                "n_outputs_": n_targets,
                "_sklearn_version": sklearn_version,
            }
        )

        estimators.append(reg)

    clf = RandomForestRegressor()
    clf.__setstate__(
        {
            "estimators_": estimators,
            "n_outputs_": n_targets,
            "_sklearn_version": sklearn_version,
        }
    )

    return clf
