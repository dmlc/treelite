# -*- coding: utf-8 -*-
"""Tests for reading/writing Protocol Buffers"""
import os

import pytest
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_compatible_toolchains, os_platform, check_predictor


@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('quantize', [True, False])
@pytest.mark.parametrize('dataset', ['mushroom', 'dermatology', 'toy_categorical'])
def test_round_trip(tmpdir, dataset, quantize, toolchain):
    """Perform round-trip tests"""
    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    pb_path = os.path.join(tmpdir, 'my.buffer')
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)

    model.export_protobuf(pb_path)
    model2 = treelite.Model.load(pb_path, model_format='protobuf')

    params = {'quantize': (1 if quantize else 0), 'parallel_comp': model.num_tree}
    model2.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    check_predictor(predictor, dataset)
