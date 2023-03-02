# -*- coding: utf-8 -*-
"""Utility functions for hypothesis-based testing"""

from sys import platform as _platform

import numpy as np
from hypothesis import assume
from hypothesis.strategies import composite, integers, just, none
from sklearn.datasets import make_classification, make_regression


def _get_limits(strategy):
    """Try to find the strategy's limits.
    Raises AttributeError if limits cannot be determined.

    Credit: Carl Simon Adorf (@csadorf)
    https://github.com/rapidsai/cuml/blob/447bded/python/cuml/testing/strategies.py
    """
    # unwrap if lazy
    strategy = getattr(strategy, "wrapped_strategy", strategy)

    try:
        yield getattr(strategy, "value")  # just(...)
    except AttributeError:
        # assume numbers strategy
        yield strategy.start
        yield strategy.stop


@composite
def standard_classification_datasets(
    draw,
    n_samples=integers(min_value=100, max_value=200),
    n_features=integers(min_value=10, max_value=20),
    *,
    n_informative=None,
    n_redundant=None,
    n_repeated=just(0),
    n_classes=just(2),
    n_clusters_per_class=just(2),
    weights=none(),
    flip_y=just(0.01),
    class_sep=just(1.0),
    hypercube=just(True),
    shift=just(0.0),
    scale=just(1.0),
    shuffle=just(True),
    random_state=None,
):
    # pylint: disable=too-many-locals
    """
    Returns a strategy to generate classification problem input datasets.
    Note:
    This function uses the sklearn.datasets.make_classification function to
    generate the classification problem from the provided search strategies.

    Credit: Carl Simon Adorf (@csadorf)
    https://github.com/rapidsai/cuml/blob/447bded/python/cuml/testing/strategies.py

    Parameters
    ----------
    draw:
        Callback function, to be used internally by Hypothesis
    n_samples: SearchStrategy[int]
        Returned arrays will have number of rows drawn from these values.
    n_features: SearchStrategy[int]
        Returned arrays will have number of columns drawn from these values.
    n_informative: SearchStrategy[int], default=none
        A search strategy for the number of informative features. If none,
        will use 10% of the actual number of features, but not less than 1
        unless the number of features is zero.
    n_redundant: SearchStrategy[int], default=none
        A search strategy for the number of redundant features. Redundant features
        will be generated as linear combinations of informative features. If none,
        will use 10% of the actual number of features, but not less than 1
        unless the number of features is zero.
    n_repeated: SearchStrategy[int], default=just(0)
        A search strategy for the number of duplicated features.
    n_classes: SearchStrategy[int], default=just(2)
        A search strategy for the number of classes in the classification problem.
    n_clusters_per_class: SearchStrategy[int], default=just(2)
        A search strategy for the number of clusters per class
    weights: SearchStrategy[array], default=none
        A search strategy for the proportions of samples assigned to each class. If
        none, always generate classification problems with balanced classes.
    flip_y: SearchStrategy[float], default=just(0.01)
        A search strategy for the fraction of samples whose class is assigned randomly.
        Larger value for this value introduces noise in the labels and make the
        classification problem harder.
    class_sep: SearchStrategy[float], default=just(1.0)
        A search strategy for the parameter class_sep.
        See sklearn.dataset.make_classification() for a detailed explanation of this
        parameter.
    hypercube: SearchStrategy[bool], default=just(True)
        A search strategy for the parameter hypercube.
        See sklearn.dataset.make_classification() for a detailed explanation of this
        parameter.
    shift: SearchStrategy[float], default=just(0.0)
        A search strategy for the parameter shift.
        See sklearn.dataset.make_classification() for a detailed explanation of this
        parameter.
    scale: SearchStrategy[float], default=just(1.0)
        A search strategy for the parameter scale.
        See sklearn.dataset.make_classification() for a detailed explanation of this
        parameter.
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
        try:
            # Try to meet:
            #   log_2(n_classes * n_clusters_per_class) <= n_informative
            n_classes_min = min(_get_limits(n_classes))
            n_clusters_per_class_min = min(_get_limits(n_clusters_per_class))
            n_informative_min = int(
                np.ceil(np.log2(n_classes_min * n_clusters_per_class_min))
            )
        except AttributeError:
            # Otherwise aim for 10% of n_features, but at least 1.
            n_informative_min = max(1, int(0.1 * n_features_))

        n_informative = just(min(n_features_, n_informative_min))
    if n_redundant is None:
        n_redundant = just(max(min(n_features_, 1), int(0.1 * n_features_)))

    # Check whether the
    #   log_2(n_classes * n_clusters_per_class) <= n_informative
    # inequality can in principle be met.
    try:
        n_classes_min = min(_get_limits(n_classes))
        n_clusters_per_class_min = min(_get_limits(n_clusters_per_class))
        n_informative_max = max(_get_limits(n_informative))
    except AttributeError:
        pass  # unable to determine limits
    else:
        if np.log2(n_classes_min * n_clusters_per_class_min) > n_informative_max:
            raise ValueError(
                "Assumptions cannot be met, the following inequality must "
                "hold: log_2(n_classes * n_clusters_per_class) "
                "<= n_informative ."
            )

    # Check base assumption concerning the composition of feature vectors.
    n_informative_ = draw(n_informative)
    n_redundant_ = draw(n_redundant)
    n_repeated_ = draw(n_repeated)
    assume(n_informative_ + n_redundant_ + n_repeated_ <= n_features_)

    # Check base assumption concerning relationship of number of clusters and
    # informative features.
    n_classes_ = draw(n_classes)
    n_clusters_per_class_ = draw(n_clusters_per_class)
    assume(np.log2(n_classes_ * n_clusters_per_class_) <= n_informative_)

    X, y = make_classification(
        n_samples=draw(n_samples),
        n_features=n_features_,
        n_informative=n_informative_,
        n_redundant=n_redundant_,
        n_repeated=n_repeated_,
        n_classes=n_classes_,
        n_clusters_per_class=n_clusters_per_class_,
        weights=draw(weights),
        flip_y=draw(flip_y),
        class_sep=draw(class_sep),
        hypercube=draw(hypercube),
        shift=draw(shift),
        scale=draw(scale),
        shuffle=draw(shuffle),
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int32)


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
    draw:
        Callback function, to be used internally by Hypothesis
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
    effective_rank:
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
