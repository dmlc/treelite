"""Tests for model builder interface"""

import numpy as np
import pytest

import treelite
from treelite import TreeliteError
from treelite.model_builder import (
    Metadata,
    ModelBuilder,
    PostProcessorFunc,
    TreeAnnotation,
)


def test_orphaned_nodes():
    """Test for orphaned nodes"""
    builder = ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=Metadata(
            num_feature=1,
            task_type="kBinaryClf",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
        postprocessor=PostProcessorFunc(name="sigmoid", sigmoid_alpha=2.0),
        base_scores=[0.0],
    )
    builder.start_tree()
    builder.start_node(0)
    builder.leaf(0.0)
    builder.end_node()
    builder.start_node(1)
    builder.leaf(1.0)
    builder.end_node()
    with pytest.raises(TreeliteError):
        builder.end_tree()


def test_invalid_node_id():
    """Test for invalid node IDs"""
    builder = ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=Metadata(
            num_feature=1,
            task_type="kBinaryClf",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
        postprocessor=PostProcessorFunc(name="sigmoid"),
        base_scores=[0.0],
    )
    builder.start_tree()
    with pytest.raises(TreeliteError):
        builder.start_node(-1)
    builder.start_node(0)

    def invalid_numerical_test(left_child_key, right_child_key):
        builder.numerical_test(
            feature_id=0,
            threshold=0.0,
            default_left=True,
            opname="<",
            left_child_key=left_child_key,
            right_child_key=right_child_key,
        )

    with pytest.raises(TreeliteError):
        invalid_numerical_test(0, 1)
    with pytest.raises(TreeliteError):
        invalid_numerical_test(2, 2)
    with pytest.raises(TreeliteError):
        invalid_numerical_test(-1, -2)
    with pytest.raises(TreeliteError):
        invalid_numerical_test(-1, 2)
    with pytest.raises(TreeliteError):
        invalid_numerical_test(2, -1)


@pytest.mark.parametrize("predict_kind", ["default", "raw", "leaf_id"])
def test_leaf_vector_rf(predict_kind):
    """Test a small random forest with leaf vector output"""
    builder = ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
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
        base_scores=[100.0, 200.0, 300.0],
    )

    def make_tree_stump(left_child_val, right_child_val):
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=0,
            threshold=0.0,
            default_left=False,
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
    dmat = np.array([[1.0], [-1.0]], dtype=np.float32)
    if predict_kind in "default":
        expected_pred = np.array([[100.0, 200.5, 300.5], [101.0, 200.0, 300.0]])
        pred = treelite.gtil.predict(model, dmat, pred_margin=False)
    elif predict_kind == "raw":
        expected_pred = np.array([[100.0, 200.5, 300.5], [101.0, 200.0, 300.0]])
        pred = treelite.gtil.predict(model, dmat, pred_margin=True)
    else:
        expected_pred = np.array([[2, 2], [1, 1]])
        pred = treelite.gtil.predict_leaf(model, dmat)
    np.testing.assert_almost_equal(pred, expected_pred, decimal=5)
