# -*- coding: utf-8 -*-
"""Tests for reading/writing Protocol Buffers"""
import os

import pytest
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_compatible_toolchains, check_predictor, does_not_raise


@pytest.mark.parametrize('code_folding_factor', [0.0, 1.0, 2.0, 3.0])
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('use_annotation', [True, False])
@pytest.mark.parametrize('dataset', ['dermatology', 'letor', 'toy_categorical'])
def test_code_folding(tmpdir, annotation, dataset, use_annotation, toolchain, code_folding_factor):
    """Test suite for testing code folding feature"""
    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    annotation_path = os.path.join(tmpdir, 'annotation.json')

    if use_annotation:
        if annotation[dataset] is None:
            pytest.skip('No training data available. Skipping annotation')
        with open(annotation_path, 'wb') as f:
            f.write(annotation[dataset])

    params = {
        'annotate_in': (annotation_path if use_annotation else 'NULL'),
        'quantize': 1,
        'parallel_comp': model.num_tree,
        'code_folding_req': code_folding_factor
    }
    # For the LightGBM model, we expect this error:
    # !t3->convert_missing_to_zero: Code folding not supported, because a categorical split is
    # supposed to convert missing values into zeros, and this is not possible with current code
    # folding implementation.
    if dataset == 'toy_categorical' and code_folding_factor < 2.0:
        expect_raises = pytest.raises(treelite.TreeliteError)
    else:
        expect_raises = does_not_raise()

    with expect_raises:
        model.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)
        predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
        check_predictor(predictor, dataset)
