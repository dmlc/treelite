"""Test field accessors"""

import numpy as np
from packaging.version import parse as parse_version

import treelite
from treelite.model_builder import (
    Metadata,
    ModelBuilder,
    PostProcessorFunc,
    TreeAnnotation,
)


def test_getter_setter():
    """Test getter and setter methods"""
    builder = ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=Metadata(
            num_feature=1,
            task_type="kMultiClf",
            average_tree_output=True,
            num_target=1,
            num_class=[3],
            leaf_vector_shape=(1, 3),
        ),
        tree_annotation=TreeAnnotation(num_tree=2, target_id=[0, 0], class_id=[-1, -1]),
        postprocessor=PostProcessorFunc(name="identity_multiclass"),
        base_scores=[0.0, 0.0, 0.0],
    )

    def make_tree_stump(left_child_val, right_child_val):
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=0,
            threshold=0.5,
            default_left=True,
            opname="<",
            left_child_key=1,
            right_child_key=2,
        )
        builder.end_node()
        builder.start_node(1)
        builder.leaf(left_child_val)
        builder.end_node()
        builder.start_node(2)
        builder.leaf(right_child_val)
        builder.end_node()
        builder.end_tree()

    make_tree_stump([1.0, 0.0, 0.0], [0.0, 0.5, 0.5])
    make_tree_stump([1.0, 0.0, 0.0], [0.0, 0.5, 0.5])

    model = builder.commit()

    header = model.get_header_accessor()
    treelite_ver = parse_version(treelite.__version__)
    expected_values = {
        "major_ver": treelite_ver.major,
        "minor_ver": treelite_ver.minor,
        "patch_ver": treelite_ver.minor,
        "threshold_type": 3,  # kFloat64
        "leaf_output_type": 3,  # kFloat64
        "num_tree": 2,
        "num_feature": 1,
        "task_type": 2,  # kMultiClf
        "average_tree_output": 1,
        "num_target": 1,
        "num_class": np.array([3]),
        "leaf_vector_shape": np.array([1, 3]),
        "target_id": np.array([0, 0]),
        "class_id": np.array([-1, -1]),
        "postprocessor": "identity_multiclass",
        "base_scores": np.array([0.0, 0.0, 0.0]),
        "num_opt_field_per_model": 0,
    }
    for k, v in expected_values.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_equal(header.get_field(k), v)
        else:
            assert header.get_field(k) == v

    num_tree = header.get_field("num_tree")[0]
    for tree_id in range(num_tree):
        tree = model.get_tree_accessor(tree_id)
        expected_values = {
            "num_nodes": 3,
            "has_categorical_split": 0,
            "node_type": np.array([1, 0, 0]),  # numerical test, leaf, leaf
            "cleft": np.array([1, -1, -1]),
            "cright": np.array([2, -1, -1]),
            "split_index": np.array([0, -1, -1]),
            "default_left": np.array([1, 0, 0]),
            "leaf_value": np.array([0, 0, 0]),
            "threshold": np.array([0.5, 0.0, 0.0]),
            "cmp": np.array([2, 0, 0]),  # kLT, kNone, kNone
            "category_list_right_child": np.array([0, 0, 0]),
            "leaf_vector": np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.5]),
            "leaf_vector_begin": np.array([0, 0, 3]),
            "leaf_vector_end": np.array([0, 3, 6]),
            "category_list": np.array([], dtype=np.int32),
            "category_list_begin": np.array([0, 0, 0]),
            "category_list_end": np.array([0, 0, 0]),
            "data_count": np.array([], dtype=np.uint64),
            "data_count_present": np.array([], dtype=np.uint8),
            "sum_hess": np.array([], dtype=np.float64),
            "sum_hess_present": np.array([], dtype=np.uint8),
            "gain": np.array([], dtype=np.float64),
            "gain_present": np.array([], dtype=np.uint8),
            "num_opt_field_per_tree": 0,
            "num_opt_field_per_node": 0,
        }
        for k, v in expected_values.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_equal(tree.get_field(k), v)
            else:
                assert tree.get_field(k) == v

    # Test setters
    header.set_field("num_feature", np.array([100], np.int32))
    assert header.get_field("num_feature") == 100
    header.set_field("postprocessor", "softmax")
    assert header.get_field("postprocessor") == "softmax"
    for tree_id in range(num_tree):
        tree = model.get_tree_accessor(tree_id)
        tree.set_field("leaf_vector", np.array([0.0] * 6))
        np.testing.assert_equal(tree.get_field("leaf_vector"), np.array([0.0] * 6))


def test_tree_editing():
    """Test ability to edit trees"""
    builder = ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=Metadata(
            num_feature=2,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[0.0],
    )
    builder.start_tree()
    builder.start_node(0)
    builder.numerical_test(
        feature_id=0,
        threshold=0.0,
        default_left=False,
        opname="<=",
        left_child_key=1,
        right_child_key=2,
    )
    builder.end_node()
    builder.start_node(1)
    builder.leaf(-1.0)
    builder.end_node()
    builder.start_node(2)
    builder.leaf(1.0)
    builder.end_node()
    builder.end_tree()

    model = builder.commit()
    np.testing.assert_array_equal(
        treelite.gtil.predict(
            model, np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ),
        np.array([[-1.0], [1.0]], dtype=np.float32),
    )

    # Change leaf values
    tree = model.get_tree_accessor(0)
    tree.set_field("leaf_value", np.array([0.0, -100.0, 100.0], dtype=np.float32))
    np.testing.assert_array_equal(
        treelite.gtil.predict(
            model, np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        ),
        np.array([[-100.0], [100.0]], dtype=np.float32),
    )

    # Change numerical test
    tree.set_field("split_index", np.array([1, -1, -1], dtype=np.int32))
    tree.set_field("threshold", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        treelite.gtil.predict(
            model, np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        ),
        np.array([[-100.0], [100.0]], dtype=np.float32),
    )

    # Add a test node
    tree.set_field("num_nodes", np.array([5], dtype=np.int32))
    tree.set_field("node_type", np.array([1, 0, 1, 0, 0], dtype=np.int8))
    tree.set_field("cleft", np.array([1, -1, 3, -1, -1], dtype=np.int32))
    tree.set_field("cright", np.array([2, -1, 4, -1, -1], dtype=np.int32))
    tree.set_field("split_index", np.array([0, -1, 1, -1, 1], dtype=np.int32))
    tree.set_field("default_left", np.array([0, 0, 0, 0, 0], dtype=np.int8))
    tree.set_field("leaf_value", np.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float32))
    tree.set_field("threshold", np.array([1.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32))
    tree.set_field("cmp", np.array([2, 0, 2, 0, 0], dtype=np.int8))
    tree.set_field("category_list_right_child", np.array([0] * 5, dtype=np.uint8))
    tree.set_field("leaf_vector_begin", np.array([0] * 5, dtype=np.uint64))
    tree.set_field("leaf_vector_end", np.array([0] * 5, dtype=np.uint64))
    tree.set_field("category_list_begin", np.array([0] * 5, dtype=np.uint64))
    tree.set_field("category_list_end", np.array([0] * 5, dtype=np.uint64))

    np.testing.assert_array_equal(
        treelite.gtil.predict(
            model,
            np.array(
                [
                    [0.5, 0.0],
                    [0.5, 5.0],
                    [2.0, 0.5],
                    [2.0, 2.5],
                ],
                dtype=np.float32,
            ),
        ),
        np.array([[1.0], [1.0], [2.0], [3.0]], dtype=np.float32),
    )
