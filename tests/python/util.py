# -*- coding: utf-8 -*-
"""Utility functions for tests"""
import os
import tempfile
from contextlib import contextmanager

import treelite

import numpy as np
from sklearn.datasets import load_svmlight_file

from .metadata import dataset_db


def load_txt(filename):
    """Get 1D array from text file"""
    if filename is None:
        return None
    content = []
    with open(filename, "r", encoding="UTF-8") as f:
        for line in f:
            content.append(float(line))
    return np.array(content, dtype=np.float32)


def has_pandas():
    """Check whether pandas is available"""
    try:
        import pandas  # pylint: disable=unused-import

        return True
    except ImportError:
        return False


@contextmanager
def does_not_raise():
    """Placeholder to indicate that a section of code is not expected to raise any exception"""
    yield


def check_gtil_output(model, dataset):
    """Check whether a predictor produces correct predictions for a given dataset"""
    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    X = X.toarray()
    X[X == 0] = np.nan
    out_margin = treelite.gtil.predict(model, X, pred_margin=True)
    out_prob = treelite.gtil.predict(model, X, pred_margin=False)
    check_predictor_output(dataset, X.shape, out_margin, out_prob)


def check_predictor_output(dataset, shape, out_margin, out_prob):
    """Check whether a predictor produces correct predictions"""
    expected_margin = load_txt(dataset_db[dataset].expected_margin)
    if dataset_db[dataset].is_multiclass:
        expected_margin = expected_margin.reshape((shape[0], -1))
    assert (
        out_margin.shape == expected_margin.shape
    ), f"out_margin.shape = {out_margin.shape}, expected_margin.shape = {expected_margin.shape}"
    np.testing.assert_almost_equal(out_margin, expected_margin, decimal=5)

    if dataset_db[dataset].expected_prob is not None:
        expected_prob = load_txt(dataset_db[dataset].expected_prob)
        if dataset_db[dataset].is_multiclass:
            expected_prob = expected_prob.reshape((shape[0], -1))
        np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@contextmanager
def TemporaryDirectory(*args, **kwargs):
    # pylint: disable=C0103
    """
    Simulate the effect of 'ignore_cleanup_errors' parameter of tempfile.TemporaryDirectory.
    The parameter is only available for Python >= 3.10.
    """
    if "PYTEST_TMPDIR" in os.environ and "dir" not in kwargs:
        kwargs["dir"] = os.environ["PYTEST_TMPDIR"]
    tmpdir = tempfile.TemporaryDirectory(*args, **kwargs)
    try:
        yield tmpdir.name
    finally:
        try:
            tmpdir.cleanup()
        except (PermissionError, NotADirectoryError):
            if _platform != "win32":
                raise
