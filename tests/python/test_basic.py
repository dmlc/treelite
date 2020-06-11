# -*- coding: utf-8 -*-
"""Suite of basic tests"""
import sys
import os
import itertools
import subprocess
from zipfile import ZipFile

import pytest
from scipy.sparse import csr_matrix
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db
from .util import os_platform, os_compatible_toolchains, does_not_raise, check_predictor


@pytest.mark.parametrize('dataset,use_annotation,parallel_comp,quantize,toolchain',
                         list(itertools.product(
                             ['mushroom', 'dermatology'], [True, False], [None, 4], [True, False],
                             os_compatible_toolchains())) +
                         [('letor', False, 713, True, os_compatible_toolchains()[0]),
                          ('toy_categorical', False, 30, True, os_compatible_toolchains()[0])])
def test_basic(tmpdir, annotation, dataset, use_annotation, quantize, parallel_comp, toolchain):
    # pylint: disable=too-many-arguments
    """Test 'ast_native' compiler"""

    if dataset == 'letor' and os_platform() == 'windows':
        pytest.xfail('export_lib() is too slow for letor on MSVC')

    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    annotation_path = os.path.join(tmpdir, 'annotation.json')

    if use_annotation:
        if annotation[dataset] is None:
            pytest.skip('No training data available. Skipping annotation')
        with open(annotation_path, 'w') as f:
            f.write(annotation[dataset])

    params = {
        'annotate_in': (annotation_path if use_annotation else 'NULL'),
        'quantize': (1 if quantize else 0),
        'parallel_comp': (parallel_comp if parallel_comp else 0)
    }
    model.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    check_predictor(predictor, dataset)


@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('use_elf', [True, False])
@pytest.mark.parametrize('dataset', ['mushroom', 'dermatology', 'letor', 'toy_categorical'])
def test_failsafe_compiler(tmpdir, dataset, use_elf, toolchain):
    """Test 'failsafe' compiler"""
    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)

    params = {'dump_array_as_elf': (1 if use_elf else 0)}

    is_linux = sys.platform.startswith('linux')
    # Expect Treelite to throw error if we try to use dump_array_as_elf on non-Linux OS
    # Also, failsafe compiler is only available for XGBoost models
    if ((not is_linux) and use_elf) or dataset_db[dataset].format != 'xgboost':
        expect_raises = pytest.raises(treelite.TreeliteError)
    else:
        expect_raises = does_not_raise()
    with expect_raises:
        model.export_lib(compiler='failsafe', toolchain=toolchain, libpath=libpath, params=params,
                         verbose=True)
        predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
        check_predictor(predictor, dataset)


@pytest.mark.skipif(os_platform() == 'windows', reason='Make unavailable on Windows')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('dataset', ['mushroom', 'dermatology', 'letor', 'toy_categorical'])
def test_srcpkg(tmpdir, dataset, toolchain):
    """Test feature to export a source tarball"""
    pkgpath = os.path.join(tmpdir, 'srcpkg.zip')
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    model.export_srcpkg(platform=os_platform(), toolchain=toolchain,
                        pkgpath=pkgpath, libname=dataset_db[dataset].libname,
                        params={'parallel_comp': 700 if dataset == 'letor' else 4}, verbose=True)
    with ZipFile(pkgpath, 'r') as zip_ref:
        zip_ref.extractall(tmpdir)
    nproc = os.cpu_count()
    subprocess.check_call(['make', '-C', dataset_db[dataset].libname, f'-j{nproc}'], cwd=tmpdir)

    libpath = os.path.join(tmpdir, dataset_db[dataset].libname)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    check_predictor(predictor, dataset)


@pytest.mark.parametrize('dataset', ['mushroom', 'dermatology', 'letor', 'toy_categorical'])
def test_srcpkg_cmake(tmpdir, dataset):  # pylint: disable=R0914
    """Test feature to export a source tarball"""
    pkgpath = os.path.join(tmpdir, 'srcpkg.zip')
    model = treelite.Model.load(dataset_db[dataset].model, model_format=dataset_db[dataset].format)
    model.export_srcpkg(platform=os_platform(), toolchain='cmake',
                        pkgpath=pkgpath, libname=dataset_db[dataset].libname,
                        params={'parallel_comp': 700 if dataset == 'letor' else 4}, verbose=True)
    with ZipFile(pkgpath, 'r') as zip_ref:
        zip_ref.extractall(tmpdir)
    build_dir = os.path.join(tmpdir, dataset_db[dataset].libname, 'build')
    os.mkdir(build_dir)
    nproc = os.cpu_count()
    subprocess.check_call(['cmake', '..'], cwd=build_dir)
    subprocess.check_call(['cmake', '--build', '.', '--config', 'Release',
                           '--parallel', str(nproc)], cwd=build_dir)

    predictor = treelite_runtime.Predictor(libpath=build_dir, verbose=True)
    check_predictor(predictor, dataset)


def test_deficient_matrix(tmpdir):
    """Test if Treelite correctly handles sparse matrix with fewer columns than the training data
    used for the model. In this case, the matrix should be padded with zeros."""
    libpath = os.path.join(tmpdir, dataset_db['mushroom'].libname + _libext())
    model = treelite.Model.load(dataset_db['mushroom'].model, model_format='xgboost')
    toolchain = os_compatible_toolchains()[0]
    model.export_lib(toolchain=toolchain, libpath=libpath, params={'quantize': 1}, verbose=True)

    X = csr_matrix(([], ([], [])), shape=(3, 3))
    batch = treelite_runtime.Batch.from_csr(X)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == 127
    predictor.predict(batch)  # should not crash


def test_too_wide_matrix(tmpdir):
    """Test if Treelite correctly handles sparse matrix with more columns than the training data
    used for the model. In this case, an exception should be thrown"""
    libpath = os.path.join(tmpdir, dataset_db['mushroom'].libname + _libext())
    model = treelite.Model.load(dataset_db['mushroom'].model, model_format='xgboost')
    toolchain = os_compatible_toolchains()[0]
    model.export_lib(toolchain=toolchain, libpath=libpath, params={'quantize': 1}, verbose=True)

    X = csr_matrix(([], ([], [])), shape=(3, 1000))
    batch = treelite_runtime.Batch.from_csr(X)
    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == 127
    pytest.raises(treelite_runtime.TreeliteRuntimeError, predictor.predict, batch)


def test_set_tree_limit():
    """Test Model.set_tree_limit"""
    model = treelite.Model.load(dataset_db['mushroom'].model, model_format='xgboost')
    assert model.num_tree == 2
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 0)
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 3)
    model.set_tree_limit(1)
    assert model.num_tree == 1

    model = treelite.Model.load(dataset_db['dermatology'].model, model_format='xgboost')
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 0)
    pytest.raises(treelite.TreeliteError, model.set_tree_limit, 200)
    assert model.num_tree == 60
    model.set_tree_limit(30)
    assert model.num_tree == 30
    model.set_tree_limit(10)
    assert model.num_tree == 10
