# -*- coding: utf-8 -*-
"""Tests for single-instance prediction"""
import os

import pytest
import numpy as np
import treelite
import treelite_runtime
from treelite.util import has_sklearn
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_compatible_toolchains, check_predictor_output


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('dataset', ['mushroom', 'dermatology', 'toy_categorical'])
def test_single_inst(tmpdir, annotation, dataset, toolchain):
    """Run end-to-end test"""
    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    annotation_path = os.path.join(tmpdir, 'annotation.json')

    if annotation[dataset] is None:
        annotation_path = None
    else:
        with open(annotation_path, 'wb') as f:
            f.write(annotation[dataset])

    params = {
        'annotate_in': (annotation_path if annotation_path else 'NULL'),
        'quantize': 1, 'parallel_comp': model.num_tree
    }
    model.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)

    from sklearn.datasets import load_svmlight_file

    X_test, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    out_prob = [[] for _ in range(4)]
    out_margin = [[] for _ in range(4)]
    for i in range(X_test.shape[0]):
        x = X_test[i, :]
        # Scipy CSR matrix
        out_prob[0].append(predictor.predict_instance(x))
        out_margin[0].append(predictor.predict_instance(x, pred_margin=True))
        # NumPy 1D array with 0 as missing value
        x = x.toarray().flatten()
        out_prob[1].append(predictor.predict_instance(x, missing=0.0))
        out_margin[1].append(predictor.predict_instance(x, missing=0.0, pred_margin=True))
        # NumPy 1D array with np.nan as missing value
        np.place(x, x == 0.0, [np.nan])
        out_prob[2].append(predictor.predict_instance(x, missing=np.nan))
        out_margin[2].append(predictor.predict_instance(x, missing=np.nan, pred_margin=True))
        # NumPy 1D array with np.nan as missing value
        # (default when `missing` parameter is unspecified)
        out_prob[3].append(predictor.predict_instance(x))
        out_margin[3].append(predictor.predict_instance(x, pred_margin=True))

    for i in range(4):
        check_predictor_output(dataset, X_test.shape,
                               out_margin=np.squeeze(np.array(out_margin[i])),
                               out_prob=np.squeeze(np.array(out_prob[i])))
