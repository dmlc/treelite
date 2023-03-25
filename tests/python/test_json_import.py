import json

import numpy as np

import treelite


def test_json_import():
    """Test ability to add and remove nodes"""
    model_obj = {
        "num_feature": 3,
        "task_type": "BinaryClfRegr",
        "average_tree_output": False,
        "task_param": {
            "output_type": "float",
            "grove_per_class": False,
            "num_class": 1,
            "leaf_vector_size": 1,
        },
        "model_param": {"pred_transform": "identity", "global_bias": 0.0},
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
                    {"node_id": 20, "leaf_value": 2.0},
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
                    {"node_id": 7, "leaf_value": 0.0},
                    {"node_id": 8, "leaf_value": 1.0},
                ],
            }
        ],
    }
    model_json = json.dumps(model_obj)

    model = treelite.Model.import_from_json(model_json)
    assert model.num_feature == 3
    assert model.num_class == 1
    assert model.num_tree == 1

    for f0 in [-0.5, 0.5, 1.5, np.nan]:
        for f1 in [0, 1, 2, 3, 4, np.nan]:
            for f2 in [-1.0, -0.5, 1.0, np.nan]:
                x = np.array([[f0, f1, f2]], dtype=np.float32)
                pred = treelite.gtil.predict(model, x)
                if f1 in [1, 2, 4] or np.isnan(f1):
                    expected_pred = 2.0
                elif f0 <= 0.5 and not np.isnan(f0):
                    expected_pred = 0.0
                else:
                    expected_pred = 1.0
                if pred != expected_pred:
                    raise ValueError(
                        f"Prediction wrong for f0={f0}, f1={f1}, f2={f2}: "
                        + f"expected_pred = {expected_pred} vs actual_pred = {pred}"
                    )
