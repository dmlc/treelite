"""Tests for model concatenation"""

import json

import pytest

import treelite
from treelite.model_builder import (
    Metadata,
    ModelBuilder,
    PostProcessorFunc,
    TreeAnnotation,
)


def test_model_concat_with_tree_stump():
    """Test model concatenation with a tree stump"""
    num_model_objs = 5

    def make_tree_stump():
        builder = ModelBuilder(
            threshold_type="float32",
            leaf_output_type="float32",
            metadata=Metadata(
                num_feature=2,
                task_type="kBinaryClf",
                average_tree_output=False,
                num_target=1,
                num_class=[1],
                leaf_vector_shape=[1, 1],
            ),
            tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
            postprocessor=PostProcessorFunc(name="sigmoid"),
            base_scores=[0.0],
        )
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=0,
            threshold=0.0,
            default_left=True,
            opname="<",
            left_child_key=1,
            right_child_key=2,
        )
        builder.end_node()
        builder.start_node(1)
        builder.leaf(1.0)
        builder.end_node()
        builder.start_node(2)
        builder.leaf(2.0)
        builder.end_node()
        builder.end_tree()
        return builder.commit()

    model_objs = [make_tree_stump() for _ in range(num_model_objs)]
    concatenated_model = treelite.Model.concatenate(model_objs)

    ref_tree_stump = make_tree_stump()
    ref_tree_stump_json = json.loads(ref_tree_stump.dump_as_json(pretty_print=False))
    concatenated_model_json = json.loads(
        concatenated_model.dump_as_json(pretty_print=False)
    )
    for attr in [
        "num_feature",
        "task_type",
        "average_tree_output",
        "num_target",
        "num_class",
        "leaf_vector_shape",
        "postprocessor",
    ]:
        assert ref_tree_stump_json[attr] == concatenated_model_json[attr]
    for attr in ["target_id", "class_id"]:
        assert (
            ref_tree_stump_json[attr] * num_model_objs == concatenated_model_json[attr]
        )
    for tree in concatenated_model_json["trees"]:
        assert ref_tree_stump_json["trees"][0] == tree


def test_model_concat_with_mismatched_tree_types():
    """Test model concatenation with a tree stump"""

    def make_tree_stump(*, threshold_type, leaf_output_type):
        builder = ModelBuilder(
            threshold_type=threshold_type,
            leaf_output_type=leaf_output_type,
            metadata=Metadata(
                num_feature=2,
                task_type="kBinaryClf",
                average_tree_output=False,
                num_target=1,
                num_class=[1],
                leaf_vector_shape=[1, 1],
            ),
            tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
            postprocessor=PostProcessorFunc(name="sigmoid"),
            base_scores=[0.0],
        )
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=0,
            threshold=0.0,
            default_left=True,
            opname="<",
            left_child_key=1,
            right_child_key=2,
        )
        builder.end_node()
        builder.start_node(1)
        builder.leaf(1.0)
        builder.end_node()
        builder.start_node(2)
        builder.leaf(2.0)
        builder.end_node()
        builder.end_tree()
        return builder.commit()

    model_objs = [
        make_tree_stump(threshold_type="float32", leaf_output_type="float32"),
        make_tree_stump(threshold_type="float64", leaf_output_type="float64"),
    ]
    with pytest.raises(
        treelite.TreeliteError,
        match=r"Model object .* has a different type than the first model object",
    ):
        treelite.Model.concatenate(model_objs)
