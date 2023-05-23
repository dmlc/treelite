# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring
"""Test whether Treelite handles invalid category values correctly"""
import numpy as np
import pytest

import treelite


@pytest.fixture(name="toy_model")
def toy_model_fixture():
    builder = treelite.ModelBuilder(num_feature=2)
    tree = treelite.ModelBuilder.Tree()
    tree[0].set_categorical_test_node(
        feature_id=1,
        left_categories=[0],
        default_left=True,
        left_child_key=1,
        right_child_key=2,
    )
    tree[1].set_leaf_node(-1.0)
    tree[2].set_leaf_node(1.0)
    tree[0].set_root()
    builder.append(tree)
    model = builder.commit()

    return model


@pytest.fixture(name="test_data")
def test_data_fixture():
    categorical_column = np.array(
        [-1, -0.6, -0.5, 0, 0.3, 0.7, 1, np.nan, np.inf, 1e10, -1e10], dtype=np.float32
    )
    dummy_column = np.zeros(categorical_column.shape[0], dtype=np.float32)
    return np.column_stack((dummy_column, categorical_column))


@pytest.fixture(name="ref_pred")
def ref_pred_fixture():
    # Negative inputs are mapped to the right child node
    # 0.3 and 0.7 are mapped to the left child node, since they get rounded toward the zero.
    # Missing value gets mapped to the left child node, since default_left=True
    # inf, 1e10, and -1e10 don't match any element of left_categories, so they get mapped to the
    # right child.
    return np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1], dtype=np.float32)


def test_gtil(toy_model, test_data, ref_pred):
    pred = treelite.gtil.predict(toy_model, test_data)
    np.testing.assert_equal(pred, ref_pred)
