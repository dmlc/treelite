# -*- coding: utf-8 -*-
"""Tests for reading/writing Protocol Buffers"""
import os
import itertools

import pytest
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_compatible_toolchains, os_platform, check_predictor


@pytest.mark.parametrize('code_folding_factor', [0.0, 1.0, 2.0, 3.0])
@pytest.mark.parametrize('dataset,toolchain',
                         list(itertools.product(['dermatology', 'toy_categorical'],
                                                os_compatible_toolchains())) +
                         [('letor', os_compatible_toolchains()[0])])
def test_code_folding(tmpdir, annotation, dataset, toolchain, code_folding_factor):
    """Test suite for testing code folding feature"""
    if dataset == 'letor' and os_platform() == 'windows':
        pytest.xfail('export_lib() is too slow for letor on MSVC')

    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    annotation_path = os.path.join(tmpdir, 'annotation.json')

    if annotation[dataset] is None:
        annotation_path = None
    else:
        with open(annotation_path, 'w') as f:
            f.write(annotation[dataset])

    params = {
        'annotate_in': (annotation_path if annotation_path else 'NULL'),
        'quantize': 1,
        'parallel_comp': model.num_tree,
        'code_folding_req': code_folding_factor
    }

    model.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    check_predictor(predictor, dataset)
