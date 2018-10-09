# -*- coding: utf-8 -*-
"""Tests for model builder interface"""
from __future__ import print_function
import unittest
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import treelite
import treelite.runtime
from util import run_pipeline_test, make_annotation, \
                 libname, os_compatible_toolchains, assert_almost_equal

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestModelBuilder(unittest.TestCase):
  def test_model_builder(self):
    builder = treelite.ModelBuilder(num_feature=127, random_forest=False,
                                    pred_transform='sigmoid')

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
    make_annotation(model=model, dtrain_path='mushroom/agaricus.train',
                    annotation_path='./annotation.json')
    for use_annotation in ['./annotation.json', None]:
      for use_quantize in [True, False]:
        run_pipeline_test(model=model,
                          dtest_path='mushroom/agaricus.test',
                          libname_fmt='./agaricus{}',
                          expected_prob_path='mushroom/agaricus.test.prob',
                          expected_margin_path='mushroom/agaricus.test.margin',
                          multiclass=False, use_annotation=use_annotation,
                          use_quantize=use_quantize,
                          use_parallel_comp=None)

  def test_model_builder2(self):
    builder = treelite.ModelBuilder(num_feature=33, random_forest=False,
                                    num_output_group=6, pred_transform='softmax',
                                    global_bias=0.5)

    # Build dermatology model
    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0587905943)
    tree[4].set_leaf_node(leaf_value=0.0906976685)
    tree[2].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_leaf_node(leaf_value=0.285522789)
    tree[6].set_leaf_node(leaf_value=0.0906976685)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=-1.6763807e-09)
    tree[8].set_leaf_node(leaf_value=-0.0560439527)
    tree[4].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.132558137)
    tree[10].set_leaf_node(leaf_value=-0.0315789469)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=10, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.264426857)
    tree[16].set_leaf_node(leaf_value=0.0631578937)
    tree[12].set_leaf_node(leaf_value=-0.042857144)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.00566037884)
    tree[14].set_leaf_node(leaf_value=-0.0539325848)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0589338616)
    tree[2].set_numerical_test_node(
      feature_id=9, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.280918717)
    tree[4].set_leaf_node(leaf_value=0.0631578937)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.0375000015)
    tree[14].set_leaf_node(leaf_value=0.0631578937)
    tree[8].set_leaf_node(leaf_value=-0.0515624993)
    tree[4].set_leaf_node(leaf_value=-0.0583710372)
    tree[2].set_numerical_test_node(
      feature_id=2, opname='<', threshold=1.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=15, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=-0.0348837189)
    tree[16].set_leaf_node(leaf_value=0.230097085)
    tree[10].set_leaf_node(leaf_value=-0.042857144)
    tree[6].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=0.0622641444)
    tree[12].set_numerical_test_node(
      feature_id=16, opname='<', threshold=1.5,
      default_left=True, left_child_key=17, right_child_key=18)
    tree[17].set_leaf_node(leaf_value=-1.6763807e-09)
    tree[18].set_numerical_test_node(
      feature_id=3, opname='<', threshold=1.5,
      default_left=True, left_child_key=19, right_child_key=20)
    tree[19].set_leaf_node(leaf_value=-0.00566037884)
    tree[20].set_leaf_node(leaf_value=-0.0554621816)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0590296499)
    tree[2].set_leaf_node(leaf_value=0.255665034)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0591240898)
    tree[2].set_leaf_node(leaf_value=0.213253006)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0580492876)
    tree[4].set_leaf_node(leaf_value=0.0831785649)
    tree[2].set_leaf_node(leaf_value=0.214440942)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.000227225872)
    tree[8].set_leaf_node(leaf_value=-0.0551713109)
    tree[4].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=-0.0314417593)
    tree[10].set_leaf_node(leaf_value=0.121289007)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=10, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.206326231)
    tree[16].set_leaf_node(leaf_value=0.0587527528)
    tree[12].set_leaf_node(leaf_value=-0.042056825)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.0051286486)
    tree[14].set_leaf_node(leaf_value=-0.0531389378)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0581933223)
    tree[2].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0549184792)
    tree[4].set_leaf_node(leaf_value=0.218241408)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.0367717631)
    tree[14].set_leaf_node(leaf_value=0.0600201152)
    tree[8].set_leaf_node(leaf_value=-0.05068909)
    tree[4].set_leaf_node(leaf_value=-0.0576147139)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0238016192)
    tree[10].set_leaf_node(leaf_value=-0.0548740439)
    tree[6].set_numerical_test_node(
      feature_id=5, opname='<', threshold=1,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=0.200441763)
    tree[12].set_leaf_node(leaf_value=-0.05085022)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0582789555)
    tree[2].set_leaf_node(leaf_value=0.201977417)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0583675168)
    tree[2].set_leaf_node(leaf_value=0.178015992)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0573389418)
    tree[4].set_leaf_node(leaf_value=0.0764855146)
    tree[2].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=18, opname='<', threshold=1.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.0148959262)
    tree[8].set_leaf_node(leaf_value=0.100288451)
    tree[6].set_leaf_node(leaf_value=0.180573165)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.000150109932)
    tree[8].set_leaf_node(leaf_value=-0.0543354563)
    tree[4].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=-0.0306716859)
    tree[10].set_leaf_node(leaf_value=0.109275043)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_numerical_test_node(
      feature_id=1, opname='<', threshold=1.5,
      default_left=True, left_child_key=17, right_child_key=18)
    tree[17].set_leaf_node(leaf_value=0.0114119714)
    tree[18].set_leaf_node(leaf_value=0.107637346)
    tree[16].set_leaf_node(leaf_value=0.176603094)
    tree[12].set_leaf_node(leaf_value=-0.041111324)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.00444722828)
    tree[14].set_leaf_node(leaf_value=-0.0522827283)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0574863814)
    tree[2].set_numerical_test_node(
      feature_id=24, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0504086725)
    tree[4].set_leaf_node(leaf_value=0.179243982)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.0359582752)
    tree[14].set_leaf_node(leaf_value=0.057139229)
    tree[8].set_leaf_node(leaf_value=-0.049840793)
    tree[4].set_leaf_node(leaf_value=-0.0568906739)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0226663705)
    tree[10].set_leaf_node(leaf_value=-0.0540132783)
    tree[6].set_numerical_test_node(
      feature_id=5, opname='<', threshold=1,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=0.166745111)
    tree[12].set_leaf_node(leaf_value=-0.0498748794)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0575630851)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.178653046)
    tree[4].set_leaf_node(leaf_value=0.0125759784)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0576536544)
    tree[2].set_leaf_node(leaf_value=0.153231919)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0567459464)
    tree[4].set_leaf_node(leaf_value=0.110998333)
    tree[2].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_leaf_node(leaf_value=0.153200045)
    tree[6].set_leaf_node(leaf_value=0.0607474744)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.000117595693)
    tree[8].set_leaf_node(leaf_value=-0.0535180643)
    tree[4].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.100398779)
    tree[10].set_leaf_node(leaf_value=-0.0303641297)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.148022339)
    tree[16].set_leaf_node(leaf_value=0.0143631762)
    tree[12].set_leaf_node(leaf_value=-0.040160954)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.00370988413)
    tree[14].set_leaf_node(leaf_value=-0.0514826663)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0568185337)
    tree[2].set_numerical_test_node(
      feature_id=28, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0158995446)
    tree[4].set_leaf_node(leaf_value=0.153841287)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.035205286)
    tree[14].set_leaf_node(leaf_value=0.0539063402)
    tree[8].set_leaf_node(leaf_value=-0.0489555411)
    tree[4].set_leaf_node(leaf_value=-0.0562078729)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=4, opname='<', threshold=1.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=-0.0318739079)
    tree[16].set_leaf_node(leaf_value=0.0676260293)
    tree[10].set_leaf_node(leaf_value=-0.0531550013)
    tree[6].set_numerical_test_node(
      feature_id=5, opname='<', threshold=1,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=0.142922416)
    tree[12].set_leaf_node(leaf_value=-0.0489839278)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.056889873)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.152286321)
    tree[4].set_leaf_node(leaf_value=0.011626726)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0569822453)
    tree[2].set_leaf_node(leaf_value=0.134278998)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0560913645)
    tree[4].set_leaf_node(leaf_value=0.101940788)
    tree[2].set_leaf_node(leaf_value=0.13119261)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.000156699578)
    tree[8].set_leaf_node(leaf_value=-0.0527174249)
    tree[4].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=-0.0296055358)
    tree[10].set_leaf_node(leaf_value=0.0916238353)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_numerical_test_node(
      feature_id=1, opname='<', threshold=1.5,
      default_left=True, left_child_key=17, right_child_key=18)
    tree[17].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=19, right_child_key=20)
    tree[19].set_leaf_node(leaf_value=0.00547444588)
    tree[20].set_leaf_node(leaf_value=0.0904819742)
    tree[18].set_leaf_node(leaf_value=0.135755911)
    tree[16].set_leaf_node(leaf_value=0.0138624636)
    tree[12].set_leaf_node(leaf_value=-0.0392000638)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.00291869789)
    tree[14].set_leaf_node(leaf_value=-0.0506643131)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0561785474)
    tree[2].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0143079786)
    tree[4].set_leaf_node(leaf_value=0.134660855)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.0343812369)
    tree[14].set_leaf_node(leaf_value=0.0508959852)
    tree[8].set_leaf_node(leaf_value=-0.0480655208)
    tree[4].set_leaf_node(leaf_value=-0.0555394776)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=1, opname='<', threshold=1.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.0659874454)
    tree[16].set_leaf_node(leaf_value=-0.0317337811)
    tree[10].set_leaf_node(leaf_value=-0.0523646288)
    tree[6].set_numerical_test_node(
      feature_id=5, opname='<', threshold=1,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=0.125240386)
    tree[12].set_leaf_node(leaf_value=-0.0480586812)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0562478416)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.133602157)
    tree[4].set_leaf_node(leaf_value=0.0103747388)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0563403554)
    tree[2].set_leaf_node(leaf_value=0.119383894)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=19, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0555937588)
    tree[4].set_leaf_node(leaf_value=0.0847212002)
    tree[2].set_leaf_node(leaf_value=0.119030036)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=-4.53396751e-05)
    tree[8].set_leaf_node(leaf_value=-0.0519176535)
    tree[4].set_numerical_test_node(
      feature_id=20, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0758128986)
    tree[10].set_leaf_node(leaf_value=0.00251008244)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=29, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.115501896)
    tree[16].set_leaf_node(leaf_value=0.0126717007)
    tree[12].set_leaf_node(leaf_value=-0.0382348187)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.00208553229)
    tree[14].set_leaf_node(leaf_value=-0.0498537868)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0555700921)
    tree[2].set_numerical_test_node(
      feature_id=28, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.013236383)
    tree[4].set_leaf_node(leaf_value=0.120270707)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.0335048102)
    tree[14].set_leaf_node(leaf_value=0.0475896709)
    tree[8].set_leaf_node(leaf_value=-0.0471977182)
    tree[4].set_leaf_node(leaf_value=-0.0549014583)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=8, opname='<', threshold=1.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.0620149337)
    tree[16].set_leaf_node(leaf_value=-0.0308423471)
    tree[10].set_leaf_node(leaf_value=-0.0515705422)
    tree[6].set_numerical_test_node(
      feature_id=5, opname='<', threshold=1,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=0.111627951)
    tree[12].set_leaf_node(leaf_value=-0.0471399762)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0556379929)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.119325638)
    tree[4].set_leaf_node(leaf_value=0.0092998175)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0557330921)
    tree[2].set_leaf_node(leaf_value=0.107890502)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0549177043)
    tree[4].set_leaf_node(leaf_value=0.0858937427)
    tree[2].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_leaf_node(leaf_value=0.108671807)
    tree[6].set_leaf_node(leaf_value=0.0449379198)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=3.66042259e-05)
    tree[8].set_leaf_node(leaf_value=-0.0511378124)
    tree[4].set_numerical_test_node(
      feature_id=20, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0703058019)
    tree[10].set_leaf_node(leaf_value=0.00258865743)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_numerical_test_node(
      feature_id=1, opname='<', threshold=1.5,
      default_left=True, left_child_key=17, right_child_key=18)
    tree[17].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=19, right_child_key=20)
    tree[19].set_leaf_node(leaf_value=0.000366851862)
    tree[20].set_leaf_node(leaf_value=0.0764373988)
    tree[18].set_leaf_node(leaf_value=0.10997247)
    tree[16].set_leaf_node(leaf_value=0.0119431121)
    tree[12].set_leaf_node(leaf_value=-0.0372664332)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.00121775095)
    tree[14].set_leaf_node(leaf_value=-0.0490503907)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0549904071)
    tree[2].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0122229299)
    tree[4].set_leaf_node(leaf_value=0.109174095)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=-0.0326999724)
    tree[12].set_leaf_node(leaf_value=0.0455024466)
    tree[8].set_leaf_node(leaf_value=-0.0463447087)
    tree[4].set_leaf_node(leaf_value=-0.0542914644)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=13, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=0.111545049)
    tree[14].set_leaf_node(leaf_value=0.0362377167)
    tree[10].set_leaf_node(leaf_value=-0.0496128574)
    tree[6].set_leaf_node(leaf_value=-0.0501813255)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0550593399)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.108341567)
    tree[4].set_leaf_node(leaf_value=0.00888466835)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0551542342)
    tree[2].set_leaf_node(leaf_value=0.0976489112)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=19, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0544739142)
    tree[4].set_leaf_node(leaf_value=0.072302945)
    tree[2].set_leaf_node(leaf_value=0.0991791785)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=2.33872252e-05)
    tree[8].set_leaf_node(leaf_value=-0.0503677316)
    tree[4].set_numerical_test_node(
      feature_id=17, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0655779094)
    tree[10].set_leaf_node(leaf_value=0.00221742759)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_numerical_test_node(
      feature_id=1, opname='<', threshold=1.5,
      default_left=True, left_child_key=17, right_child_key=18)
    tree[17].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=19, right_child_key=20)
    tree[19].set_leaf_node(leaf_value=-0.000183046126)
    tree[20].set_leaf_node(leaf_value=0.0714399219)
    tree[18].set_leaf_node(leaf_value=0.100781284)
    tree[16].set_leaf_node(leaf_value=0.0116898036)
    tree[12].set_leaf_node(leaf_value=-0.0362947844)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=-0.000583527843)
    tree[14].set_leaf_node(leaf_value=-0.0482258946)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.054444056)
    tree[2].set_numerical_test_node(
      feature_id=28, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0110569848)
    tree[4].set_leaf_node(leaf_value=0.100381367)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=-0.0318585858)
    tree[12].set_leaf_node(leaf_value=0.0426461659)
    tree[8].set_leaf_node(leaf_value=-0.0455001295)
    tree[4].set_leaf_node(leaf_value=-0.0537060164)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=13, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=0.101871371)
    tree[14].set_leaf_node(leaf_value=0.0334801935)
    tree[10].set_leaf_node(leaf_value=-0.0487776436)
    tree[6].set_leaf_node(leaf_value=-0.0493848994)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0545130074)
    tree[2].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0994988531)
    tree[4].set_leaf_node(leaf_value=0.00806681532)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0546116605)
    tree[2].set_numerical_test_node(
      feature_id=17, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0984470174)
    tree[4].set_leaf_node(leaf_value=0.0319658034)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=21, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0538570397)
    tree[4].set_leaf_node(leaf_value=0.0738569945)
    tree[2].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_leaf_node(leaf_value=0.0926994756)
    tree[6].set_leaf_node(leaf_value=0.0392167829)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.000280787208)
    tree[8].set_leaf_node(leaf_value=-0.0496028587)
    tree[4].set_numerical_test_node(
      feature_id=20, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0622923337)
    tree[10].set_leaf_node(leaf_value=0.0018609073)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=29, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_leaf_node(leaf_value=0.0888965875)
    tree[16].set_leaf_node(leaf_value=0.0108379442)
    tree[12].set_leaf_node(leaf_value=-0.0353214256)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=6.99278025e-05)
    tree[14].set_leaf_node(leaf_value=-0.0474047288)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0539219081)
    tree[2].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0105951307)
    tree[4].set_leaf_node(leaf_value=0.0931921825)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=-0.0310331918)
    tree[12].set_leaf_node(leaf_value=0.0400059558)
    tree[8].set_leaf_node(leaf_value=-0.0446291976)
    tree[4].set_leaf_node(leaf_value=-0.0531485341)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=13, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=0.0940201879)
    tree[14].set_leaf_node(leaf_value=0.0309093092)
    tree[10].set_leaf_node(leaf_value=-0.0479226783)
    tree[6].set_leaf_node(leaf_value=-0.0486013182)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0539949834)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.092454873)
    tree[4].set_leaf_node(leaf_value=0.00714519015)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0540942661)
    tree[2].set_numerical_test_node(
      feature_id=17, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0910510495)
    tree[4].set_leaf_node(leaf_value=0.0300962757)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=21, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=19, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=-0.0534542017)
    tree[4].set_leaf_node(leaf_value=0.0630955249)
    tree[2].set_leaf_node(leaf_value=0.086135909)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=27, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=12, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=31, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_leaf_node(leaf_value=0.000330694107)
    tree[8].set_leaf_node(leaf_value=-0.0488443002)
    tree[4].set_numerical_test_node(
      feature_id=17, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_leaf_node(leaf_value=0.0583632067)
    tree[10].set_leaf_node(leaf_value=0.0015067599)
    tree[2].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=11, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_numerical_test_node(
      feature_id=6, opname='<', threshold=0.5,
      default_left=True, left_child_key=15, right_child_key=16)
    tree[15].set_numerical_test_node(
      feature_id=1, opname='<', threshold=1.5,
      default_left=True, left_child_key=17, right_child_key=18)
    tree[17].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=19, right_child_key=20)
    tree[19].set_leaf_node(leaf_value=-0.00412822422)
    tree[20].set_leaf_node(leaf_value=0.0624891929)
    tree[18].set_leaf_node(leaf_value=0.0875311866)
    tree[16].set_leaf_node(leaf_value=0.0106048854)
    tree[12].set_leaf_node(leaf_value=-0.0343477204)
    tree[6].set_numerical_test_node(
      feature_id=15, opname='<', threshold=1.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=0.000736902177)
    tree[14].set_leaf_node(leaf_value=-0.0465852581)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=32, opname='<', threshold=1.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0534253083)
    tree[2].set_numerical_test_node(
      feature_id=28, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.00922845583)
    tree[4].set_leaf_node(leaf_value=0.0873670429)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=4, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_numerical_test_node(
      feature_id=0, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_numerical_test_node(
      feature_id=3, opname='<', threshold=0.5,
      default_left=True, left_child_key=7, right_child_key=8)
    tree[7].set_numerical_test_node(
      feature_id=27, opname='<', threshold=0.5,
      default_left=True, left_child_key=11, right_child_key=12)
    tree[11].set_leaf_node(leaf_value=-0.0301814582)
    tree[12].set_leaf_node(leaf_value=0.0384008698)
    tree[8].set_leaf_node(leaf_value=-0.0437796563)
    tree[4].set_leaf_node(leaf_value=-0.0526101775)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=5, right_child_key=6)
    tree[5].set_numerical_test_node(
      feature_id=19, opname='<', threshold=0.5,
      default_left=True, left_child_key=9, right_child_key=10)
    tree[9].set_numerical_test_node(
      feature_id=13, opname='<', threshold=0.5,
      default_left=True, left_child_key=13, right_child_key=14)
    tree[13].set_leaf_node(leaf_value=0.0875401646)
    tree[14].set_leaf_node(leaf_value=0.0285075735)
    tree[10].set_leaf_node(leaf_value=-0.0471032001)
    tree[6].set_leaf_node(leaf_value=-0.0478212647)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=14, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.053501375)
    tree[2].set_numerical_test_node(
      feature_id=32, opname='<', threshold=0.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0865296647)
    tree[4].set_leaf_node(leaf_value=0.00686053885)
    tree[0].set_root()
    builder.append(tree)

    tree = treelite.ModelBuilder.Tree()
    tree[0].set_numerical_test_node(
      feature_id=30, opname='<', threshold=0.5,
      default_left=True, left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(leaf_value=-0.0536041856)
    tree[2].set_numerical_test_node(
      feature_id=17, opname='<', threshold=1.5,
      default_left=True, left_child_key=3, right_child_key=4)
    tree[3].set_leaf_node(leaf_value=0.0851988941)
    tree[4].set_leaf_node(leaf_value=0.0278433915)
    tree[0].set_root()
    builder.append(tree)

    model = builder.commit()
    make_annotation(model=model, dtrain_path='dermatology/dermatology.train',
                    annotation_path='./annotation.json')
    for use_annotation in ['./annotation.json', None]:
      for use_quantize in [True, False]:
        run_pipeline_test(model=model,
                          dtest_path='dermatology/dermatology.test',
                          libname_fmt='./dermatology{}',
                          expected_prob_path='dermatology/dermatology.test.prob',
                          expected_margin_path='dermatology/dermatology.test.margin',
                          multiclass=True, use_annotation=use_annotation,
                          use_quantize=use_quantize, use_parallel_comp=None)

  def test_model_builder3(self):
    """Test programmatic model construction using scikit-learn random forest"""
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

    def process_node(treelite_tree, sklearn_tree, nodeid):
      if sklearn_tree.children_left[nodeid] == -1:  # leaf node
        leaf_count = sklearn_tree.value[nodeid].squeeze()
        prob_distribution = leaf_count / leaf_count.sum()
        treelite_tree[nodeid].set_leaf_node(prob_distribution)
      else:  # test node
        treelite_tree[nodeid].set_numerical_test_node(
          feature_id=sklearn_tree.feature[nodeid],
          opname='<=',
          threshold=sklearn_tree.threshold[nodeid],
          default_left=True,
          left_child_key=sklearn_tree.children_left[nodeid],
          right_child_key=sklearn_tree.children_right[nodeid])

    def process_tree(sklearn_tree):
      treelite_tree = treelite.ModelBuilder.Tree()
      treelite_tree[0].set_root()
      for nodeid in range(sklearn_tree.node_count):
        process_node(treelite_tree, sklearn_tree, nodeid)
      return treelite_tree

    builder = treelite.ModelBuilder(num_feature=clf.n_features_,
                                    num_output_group=clf.n_classes_,
                                    random_forest=True,
                                    pred_transform='identity_multiclass')
    for i in range(clf.n_estimators):
      builder.append(process_tree(clf.estimators_[i].tree_))

    model = builder.commit()

    dtrain = treelite.DMatrix(X)
    annotator = treelite.Annotator()
    annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
    annotator.save(path='./annotation.json')

    libpath = libname('./iris{}')
    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params={'annotate_in': './annotation.json'}, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      batch = treelite.runtime.Batch.from_npy2d(X)
      out_prob = predictor.predict(batch)
      assert_almost_equal(out_prob, expected_prob)

    # Test round-trip with Protobuf
    model.export_protobuf('./my.buffer')
    model = treelite.Model.load('./my.buffer', 'protobuf')
    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params={'annotate_in': './annotation.json'}, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      batch = treelite.runtime.Batch.from_npy2d(X)
      out_prob = predictor.predict(batch)
      assert_almost_equal(out_prob, expected_prob)

  def test_node_insert_delete(self):
    """Test ability to add and remove nodes"""
    builder = treelite.ModelBuilder(num_feature=3)
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
    libpath = libname('./libtest{}')
    model.export_lib(toolchain='gcc', libpath=libpath, verbose=True)
    predictor = treelite.runtime.Predictor(libpath=libpath)
    for f0 in [-0.5, 0.5, 1.5, np.nan]:
      for f1 in [0, 1, 2, 3, 4, np.nan]:
        for f2 in [-1.0, -0.5, 1.0, np.nan]:
          x = np.array([f0, f1, f2])
          pred = predictor.predict_instance(x)
          if f1 in [1, 2, 4] or np.isnan(f1):
            expected_pred = 2.0
          elif f0 <= 0.5 and not np.isnan(f0):
            expected_pred = 0.0
          else:
            expected_pred = 1.0
          assert pred == expected_pred, \
            'Prediction wrong for f0={}, f1={}, f2={}: '.format(f0, f1, f2) + \
            'expected_pred = {} vs actual_pred = {}'.format(expected_pred, pred)
