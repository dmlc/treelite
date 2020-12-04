# -*- coding: utf-8 -*-
"""Tests for model builder interface"""
import os

import pytest
import numpy as np
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_compatible_toolchains, check_predictor


@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('quantize', [True, False])
@pytest.mark.parametrize('use_annotation', [True, False])
def test_model_builder(tmpdir, use_annotation, quantize, toolchain):
    """A simple model"""
    num_feature = 127
    pred_transform = 'sigmoid'
    builder = treelite.ModelBuilder(num_feature=num_feature, average_tree_output=False,
                                    pred_transform=pred_transform)

    # Build mushroom model
    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
        feature_id=29, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
        feature_id=56, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
        feature_id=60, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=1.89899647)
    tree[8].set_leaf_node(leaf_value=-1.94736838)
    tree[4].set_numerical_test_node(
        feature_id=21, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=1.78378379)
    tree[10].set_leaf_node(leaf_value=-1.98135197)
    tree[2].set_numerical_test_node(
        feature_id=109, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
        feature_id=67, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=-1.9854598)
    tree[12].set_leaf_node(leaf_value=0.938775539)
    tree[6].set_leaf_node(leaf_value=1.87096775)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
        feature_id=29, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
        feature_id=21, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=1.14607906)
    tree[4].set_numerical_test_node(
        feature_id=36, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=-6.87994671)
    tree[8].set_leaf_node(leaf_value=-0.10659159)
    tree[2].set_numerical_test_node(
        feature_id=109, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
        feature_id=39, opname='<', threshold=-9.53674316e-07, default_left=True,
        left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=-0.0930657759)
    tree[10].set_leaf_node(leaf_value=-1.15261209)
    tree[6].set_leaf_node(leaf_value=1.00423074)
    tree[0].set_root()
    builder.append(tree)

    model = builder.commit()
    assert model.num_feature == num_feature
    assert model.num_class == 1
    assert model.num_tree == 2

    annotation_path = os.path.join(tmpdir, 'annotation.json')
    if use_annotation:
        dtrain = treelite_runtime.DMatrix(dataset_db['mushroom'].dtrain, dtype='float32')
        annotator = treelite.Annotator()
        annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
        annotator.save(path=annotation_path)

    params = {
        'annotate_in': (annotation_path if use_annotation else 'NULL'),
        'quantize': (1 if quantize else 0),
        'parallel_comp': model.num_tree
    }
    libpath = os.path.join(tmpdir, 'mushroom' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    check_predictor(predictor, 'mushroom')


@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
def test_node_insert_delete(tmpdir, toolchain):
    """Test ability to add and remove nodes"""
    num_feature = 3
    builder = treelite.ModelBuilder(num_feature=num_feature)
    builder.append(treelite.ModelBuilder.Tree())
    builder[0][1].set_root()
    builder[0][1].set_numerical_test_node(
        feature_id=2, opname='<', threshold=-0.5, default_left=True,
        left_child_key=5, right_child_key=10)
    builder[0][5].set_leaf_node(-1)
    builder[0][10].set_numerical_test_node(
        feature_id=0, opname='<=', threshold=0.5, default_left=False,
        left_child_key=7, right_child_key=8)
    builder[0][7].set_leaf_node(0.0)
    builder[0][8].set_leaf_node(1.0)
    del builder[0][1]
    del builder[0][5]
    builder[0][5].set_categorical_test_node(
        feature_id=1, left_categories=[1, 2, 4], default_left=True,
        left_child_key=20, right_child_key=10)
    builder[0][20].set_leaf_node(2.0)
    builder[0][5].set_root()

    model = builder.commit()
    assert model.num_feature == num_feature
    assert model.num_class == 1
    assert model.num_tree == 1

    libpath = os.path.join(tmpdir, 'libtest' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath)
    assert predictor.num_feature == num_feature
    assert predictor.num_class == 1
    assert predictor.pred_transform == 'identity'
    assert predictor.global_bias == 0.0
    assert predictor.sigmoid_alpha == 1.0
    for f0 in [-0.5, 0.5, 1.5, np.nan]:
        for f1 in [0, 1, 2, 3, 4, np.nan]:
            for f2 in [-1.0, -0.5, 1.0, np.nan]:
                x = np.array([[f0, f1, f2]])
                dmat = treelite_runtime.DMatrix(x, dtype='float32')
                pred = predictor.predict(dmat)
                if f1 in [1, 2, 4] or np.isnan(f1):
                    expected_pred = 2.0
                elif f0 <= 0.5 and not np.isnan(f0):
                    expected_pred = 0.0
                else:
                    expected_pred = 1.0
                if pred != expected_pred:
                    raise ValueError(f'Prediction wrong for f0={f0}, f1={f1}, f2={f2}: ' +
                                     f'expected_pred = {expected_pred} vs actual_pred = {pred}')
