# -*- coding: utf-8 -*-
"""Tests for model concatenation"""

import json

import pytest

import treelite


def test_model_concat_with_tree_stump():
    """Test model concatenation with a tree stump"""
    num_feature = 2
    num_model_objs = 5
    pred_transform = "sigmoid"

    def make_tree_stump():
        builder = treelite.ModelBuilder(
            num_feature=num_feature,
            average_tree_output=False,
            pred_transform=pred_transform,
            threshold_type="float32",
            leaf_output_type="float32",
        )
        tree = treelite.ModelBuilder.Tree(
            threshold_type="float32", leaf_output_type="float32"
        )
        tree[0].set_numerical_test_node(
            feature_id=0,
            opname="<",
            threshold=0.0,
            default_left=True,
            left_child_key=1,
            right_child_key=2,
            threshold_type="float32",
        )
        tree[1].set_leaf_node(leaf_value=1.0, leaf_value_type="float32")
        tree[2].set_leaf_node(leaf_value=2.0, leaf_value_type="float32")
        tree[0].set_root()
        builder.append(tree)
        return builder.commit()

    model_objs = [make_tree_stump() for _ in range(num_model_objs)]
    concatenated_model = treelite.Model.concatenate(model_objs)

    ref_tree_stump = make_tree_stump()
    ref_tree_stump_json = json.loads(ref_tree_stump.dump_as_json(pretty_print=False))
    concatenated_model_json = json.loads(
        concatenated_model.dump_as_json(pretty_print=False)
    )
    for attr in ["num_feature", "task_type", "average_tree_output", "model_param"]:
        assert ref_tree_stump_json[attr] == concatenated_model_json[attr]
    for tree in concatenated_model_json["trees"]:
        assert ref_tree_stump_json["trees"][0] == tree


def test_model_concat_with_mismatched_tree_types():
    """Test model concatenation with a tree stump"""
    num_feature = 2
    pred_transform = "sigmoid"

    def make_tree_stump(*, threshold_type, leaf_output_type):
        builder = treelite.ModelBuilder(
            num_feature=num_feature,
            average_tree_output=False,
            pred_transform=pred_transform,
            threshold_type=threshold_type,
            leaf_output_type=leaf_output_type,
        )
        tree = treelite.ModelBuilder.Tree(
            threshold_type=threshold_type, leaf_output_type=leaf_output_type
        )
        tree[0].set_numerical_test_node(
            feature_id=0,
            opname="<",
            threshold=0.0,
            default_left=True,
            left_child_key=1,
            right_child_key=2,
            threshold_type=threshold_type,
        )
        tree[1].set_leaf_node(leaf_value=1.0, leaf_value_type=leaf_output_type)
        tree[2].set_leaf_node(leaf_value=2.0, leaf_value_type=leaf_output_type)
        tree[0].set_root()
        builder.append(tree)
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
