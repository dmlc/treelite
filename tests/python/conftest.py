# -*- coding: utf-8 -*-
"""Pytest fixtures to initialize tests"""
import os
import tempfile

import pytest
import treelite
import treelite_runtime
from sklearn.datasets import load_svmlight_file
from .metadata import dataset_db


@pytest.fixture(scope='session')
def annotation():
    """Pre-computed branch annotation information for example datasets"""
    with tempfile.TemporaryDirectory(dir='.') as tmpdir:
        def compute_annotation(dataset):
            model = treelite.Model.load(dataset_db[dataset].model,
                                        model_format=dataset_db[dataset].format)
            if dataset_db[dataset].dtrain is None:
                return None
            dtrain = treelite_runtime.DMatrix(
                load_svmlight_file(dataset_db[dataset].dtrain, zero_based=True)[0])
            annotator = treelite.Annotator()
            annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
            annotation_path = os.path.join(tmpdir, f'{dataset}.json')
            annotator.save(annotation_path)
            with open(annotation_path, 'r', encoding='UTF-8') as f:
                return f.read()
        annotation_db = {k: compute_annotation(k) for k in dataset_db}
    return annotation_db
