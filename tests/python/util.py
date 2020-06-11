# -*- coding: utf-8 -*-
"""Utility functions for tests"""
import os
from sys import platform as _platform
from contextlib import contextmanager

import numpy as np
import treelite
import treelite_runtime
from treelite.contrib import _libext
from .metadata import dataset_db


def load_txt(filename):
    """Get 1D array from text file"""
    if filename is None:
        return None
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(float(line))
    return np.array(content)


def os_compatible_toolchains():
    """Get the list of C compilers to test with the current OS"""
    if _platform == 'darwin':
        gcc = os.environ.get('GCC_PATH', 'gcc')
        toolchains = [gcc]
    elif _platform == 'win32':
        toolchains = ['msvc']
    else:
        toolchains = ['gcc', 'clang']
    return toolchains


def os_platform():
    """Detect OS that's running this program"""
    if _platform == 'darwin':
        return 'osx'
    if _platform in ['win32', 'cygwin']:
        return 'windows'
    return 'unix'


def libname(fmt):
    """Format name for a shared library, using appropriate file extension"""
    return fmt.format(_libext())


@contextmanager
def does_not_raise():
    """Placeholder to indicate that a section of code is not expected to raise any exception"""
    yield


def check_predictor(predictor, dataset):
    """Check whether a predictor produces correct predictions for a given dataset"""
    dtest = treelite.DMatrix(dataset_db[dataset].dtest)
    batch = treelite_runtime.Batch.from_csr(dtest)
    out_margin = predictor.predict(batch, pred_margin=True)
    out_prob = predictor.predict(batch)
    check_predictor_output(dataset, dtest.shape, out_margin, out_prob)


def check_predictor_output(dataset, shape, out_margin, out_prob):
    """Check whether a predictor produces correct predictions"""
    expected_margin = load_txt(dataset_db[dataset].expected_margin)
    if dataset_db[dataset].is_multiclass:
        expected_margin = expected_margin.reshape((shape[0], -1))
    np.testing.assert_almost_equal(out_margin, expected_margin, decimal=5)

    if dataset_db[dataset].expected_prob is not None:
        expected_prob = load_txt(dataset_db[dataset].expected_prob)
        if dataset_db[dataset].is_multiclass:
            expected_prob = expected_prob.reshape((shape[0], -1))
        np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)
