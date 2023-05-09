# -*- coding: utf-8 -*-
"""Tests for rich model query functions"""

import collections
import os

import pytest
import treelite
from .metadata import dataset_db

ModelFact = collections.namedtuple(
    'ModelFact',
    'num_tree num_feature num_class pred_transform global_bias sigmoid_alpha ratio_c '
    'threshold_type leaf_output_type')
_model_facts = {
    'mushroom': ModelFact(2, 127, 1, 'sigmoid', 0.0, 1.0, 1.0, 'float32', 'float32'),
    'dermatology': ModelFact(60, 33, 6, 'softmax', 0.5, 1.0, 1.0, 'float32', 'float32'),
    'letor': ModelFact(713, 47, 1, 'identity', 0.5, 1.0, 1.0, 'float32', 'float32'),
    'toy_categorical': ModelFact(30, 2, 1, 'identity', 0.0, 1.0, 1.0, 'float64', 'float64'),
    'sparse_categorical': ModelFact(1, 5057, 1, 'sigmoid', 0.0, 1.0, 1.0, 'float64', 'float64')
}


@pytest.mark.parametrize(
    'dataset', ['mushroom', 'dermatology', 'letor', 'toy_categorical', 'sparse_categorical'])
def test_model_query(tmpdir, dataset):
    """Test all query functions for every example model"""

    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    assert model.num_feature == _model_facts[dataset].num_feature
    assert model.num_class == _model_facts[dataset].num_class
    assert model.num_tree == _model_facts[dataset].num_tree
