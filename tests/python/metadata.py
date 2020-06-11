# -*- coding: utf-8 -*-
"""Metadata for datasets and models used for testing"""

import collections
import os

_current_dir = os.path.dirname(__file__)
_dpath = os.path.abspath(os.path.join(_current_dir, os.path.pardir, 'examples'))

Dataset = collections.namedtuple(
    'Dataset', 'model format dtrain dtest libname expected_prob expected_margin is_multiclass')

_dataset_db = {
    'mushroom': Dataset(model='mushroom.model', format='xgboost', dtrain='agaricus.train',
                        dtest='agaricus.test', libname='agaricus',
                        expected_prob='agaricus.test.prob', expected_margin='agaricus.test.margin',
                        is_multiclass=False),
    'dermatology': Dataset(model='dermatology.model', format='xgboost', dtrain='dermatology.train',
                           dtest='dermatology.test', libname='dermatology',
                           expected_prob='dermatology.test.prob',
                           expected_margin='dermatology.test.margin', is_multiclass=True),
    'letor': Dataset(model='mq2008.model', format='xgboost', dtrain='mq2008.train',
                     dtest='mq2008.test', libname='letor', expected_prob=None,
                     expected_margin='mq2008.test.pred', is_multiclass=False),
    'toy_categorical': Dataset(model='toy_categorical_model.txt', format='lightgbm', dtrain=None,
                               dtest='toy_categorical.test', libname='toycat', expected_prob=None,
                               expected_margin='toy_categorical.test.pred', is_multiclass=False),
    'sparse_categorical': Dataset(model='sparse_categorical_model.txt', format='lightgbm',
                                  dtrain=None, dtest='sparse_categorical.test', libname='sparsecat',
                                  expected_prob=None,
                                  expected_margin='sparse_categorical.test.pred',
                                  is_multiclass=False)
}


def _qualify_path(prefix, path):
    if path is None:
        return None
    return os.path.join(_dpath, prefix, path)


dataset_db = {
    k: v._replace(model=_qualify_path(k, v.model), dtrain=_qualify_path(k, v.dtrain),
                  dtest=_qualify_path(k, v.dtest), expected_prob=_qualify_path(k, v.expected_prob),
                  expected_margin=_qualify_path(k, v.expected_margin))
    for k, v in _dataset_db.items()
}
