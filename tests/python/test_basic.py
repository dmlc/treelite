# -*- coding: utf-8 -*-
"""Suite of basic tests"""
import pytest
import treelite
from .metadata import dataset_db


def test_set_tree_limit():
    """Test Model.set_tree_limit"""
    model = treelite.Model.load(dataset_db['mushroom'].model, model_format='xgboost')
    assert model.num_tree == 2
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 0)
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 3)
    model.set_tree_limit(1)
    assert model.num_tree == 1

    model = treelite.Model.load(dataset_db['dermatology'].model, model_format='xgboost')
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 0)
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 200)
    assert model.num_tree == 60
    model.set_tree_limit(30)
    assert model.num_tree == 30
    model.set_tree_limit(10)
    assert model.num_tree == 10
