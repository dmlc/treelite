import json

import numpy as np
import pytest

import treelite


@pytest.mark.parametrize("use_vector_leaf", [True, False])
def test_json_import(use_vector_leaf):
    """Test JSON import feature"""
    if use_vector_leaf:
        task_type = "MultiClfProbDistLeaf"
        num_class = 3
        leaf_vector_size = 3
        pred_transform = "softmax"
    else:
        task_type = "BinaryClfRegr"
        num_class = 1
        leaf_vector_size = 1
        pred_transform = "sigmoid"

    def get_leaf_value(leaf_val):
        if use_vector_leaf:
            return [leaf_val] * leaf_vector_size
        return leaf_val

    model_obj = {
        "num_feature": 3,
        "task_type": task_type,
        "average_tree_output": False,
        "task_param": {
            "output_type": "float",
            "grove_per_class": False,
            "num_class": num_class,
            "leaf_vector_size": leaf_vector_size,
        },
        "model_param": {"pred_transform": pred_transform, "global_bias": 0.0},
        "trees": [
            {
                "root_id": 5,
                "nodes": [
                    {
                        "node_id": 5,
                        "split_feature_id": 1,
                        "default_left": True,
                        "split_type": "categorical",
                        "categories_list": [1, 2, 4],
                        "categories_list_right_child": False,
                        "left_child": 20,
                        "right_child": 10,
                    },
                    {"node_id": 20, "leaf_value": get_leaf_value(2.0)},
                    {
                        "node_id": 10,
                        "split_feature_id": 0,
                        "default_left": False,
                        "split_type": "numerical",
                        "comparison_op": "<=",
                        "threshold": 0.5,
                        "left_child": 7,
                        "right_child": 8,
                    },
                    {"node_id": 7, "leaf_value": get_leaf_value(0.0)},
                    {"node_id": 8, "leaf_value": get_leaf_value(1.0)},
                ],
            }
        ],
    }
    model_json = json.dumps(model_obj)

    model = treelite.Model.import_from_json(model_json)
    assert model.num_feature == 3
    assert model.num_class == num_class
    assert model.num_tree == 1

    for f0 in [-0.5, 0.5, 1.5, np.nan]:
        for f1 in [0, 1, 2, 3, 4, np.nan]:
            for f2 in [-1.0, -0.5, 1.0, np.nan]:
                x = np.array([[f0, f1, f2]], dtype=np.float32)
                pred = treelite.gtil.predict(model, x)
                if f1 in [1, 2, 4] or np.isnan(f1):
                    expected_pred = get_leaf_value(2.0)
                elif f0 <= 0.5 and not np.isnan(f0):
                    expected_pred = get_leaf_value(0.0)
                else:
                    expected_pred = get_leaf_value(1.0)
                if np.array_equal(pred, expected_pred):
                    raise ValueError(
                        f"Prediction wrong for f0={f0}, f1={f1}, f2={f2}: "
                        + f"expected_pred = {expected_pred} vs actual_pred = {pred}"
                    )
