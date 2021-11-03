# -*- coding: utf-8 -*-
"""Tests for rich model query functions"""

import collections
import os

import pytest
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_platform, os_compatible_toolchains

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
    if dataset == 'sparse_categorical':
        if os_platform() == 'windows':
            pytest.xfail('MSVC cannot handle long if conditional')
        elif os_platform() == 'osx':
            pytest.xfail('Apple Clang cannot handle long if conditional')
    if dataset == 'letor' and os_platform() == 'windows':
        pytest.xfail('export_lib() is too slow for letor on MSVC')

    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    assert model.num_feature == _model_facts[dataset].num_feature
    assert model.num_class == _model_facts[dataset].num_class
    assert model.num_tree == _model_facts[dataset].num_tree

    toolchain = os_compatible_toolchains()[0]
    model.export_lib(toolchain=toolchain, libpath=libpath,
                     params={'quantize': 1, 'parallel_comp': model.num_tree}, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == _model_facts[dataset].num_feature
    assert predictor.num_class == _model_facts[dataset].num_class
    assert predictor.pred_transform == _model_facts[dataset].pred_transform
    assert predictor.global_bias == _model_facts[dataset].global_bias
    assert predictor.sigmoid_alpha == _model_facts[dataset].sigmoid_alpha
    assert predictor.ratio_c == _model_facts[dataset].ratio_c
    assert predictor.threshold_type == _model_facts[dataset].threshold_type
    assert predictor.leaf_output_type == _model_facts[dataset].leaf_output_type
