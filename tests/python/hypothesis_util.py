# -*- coding: utf-8 -*-
"""Utility functions for hypothesis-based testing"""

import os
from sys import platform as _platform

import numpy as np
from hypothesis.strategies import composite, integers, just, none
from sklearn.datasets import make_regression


@composite
def standard_regression_datasets(
    draw,
    n_samples=integers(min_value=100, max_value=200),
    n_features=integers(min_value=100, max_value=200),
    *,
    n_informative=None,
    n_targets=just(1),
    bias=just(0.0),
    effective_rank=none(),
    tail_strength=just(0.5),
    noise=just(0.0),
    shuffle=just(True),
    random_state=None,
):
    """
    Returns a strategy to generate regression problem input datasets.
    Note:
    This function uses the sklearn.datasets.make_regression function to
    generate the regression problem from the provided search strategies.

    Credit: Carl Simon Adorf (@csadorf)
    https://github.com/rapidsai/cuml/blob/447bded/python/cuml/testing/strategies.py

    Parameters
    ----------
    n_samples: SearchStrategy[int]
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int]
        Returned arrays will have number of columns drawn from these values.
    n_informative: SearchStrategy[int], default=none
        A search strategy for the number of informative features. If none,
        will use 10% of the actual number of features, but not less than 1
        unless the number of features is zero.
    n_targets: SearchStrategy[int], default=just(1)
        A search strategy for the number of targets, that means the number of
        columns of the returned y output array.
    bias: SearchStrategy[float], default=just(0.0)
        A search strategy for the bias term.
    effective_rank=none()
        If not None, a search strategy for the effective rank of the input data
        for the regression problem. See sklearn.dataset.make_regression() for a
        detailed explanation of this parameter.
    tail_strength: SearchStrategy[float], default=just(0.5)
        See sklearn.dataset.make_regression() for a detailed explanation of
        this parameter.
    noise: SearchStrategy[float], default=just(0.0)
        A search strategy for the standard deviation of the gaussian noise.
    shuffle: SearchStrategy[bool], default=just(True)
        A boolean search strategy to determine whether samples and features
        are shuffled.
    random_state: int, RandomState instance or None, default=None
        Pass a random state or integer to determine the random number
        generation for data set generation.
    Returns
    -------
    (X, y):  SearchStrategy[array], SearchStrategy[array]
        A tuple of search strategies for arrays subject to the constraints of
        the provided parameters.
    """
    n_features_ = draw(n_features)
    if n_informative is None:
        n_informative = just(max(min(n_features_, 1), int(0.1 * n_features_)))
    X, y = make_regression(
        n_samples=draw(n_samples),
        n_features=n_features_,
        n_informative=draw(n_informative),
        n_targets=draw(n_targets),
        bias=draw(bias),
        effective_rank=draw(effective_rank),
        tail_strength=draw(tail_strength),
        noise=draw(noise),
        shuffle=draw(shuffle),
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.float32)


def standard_settings():
    """Default hypotheiss settings. Set a smaller max_examples on Windows"""
    kwargs = {
        "deadline": None,
        "max_examples": 20,
        "print_blob": True,
    }
    if _platform == "win32":
        kwargs["max_examples"] = 3
    return kwargs
