"""Utility functions for loading IsolationForest models"""

import numpy as np
from scipy.special import psi


def harmonic(number):
    """Calculates the n-th harmonic number"""
    return psi(number + 1) + np.euler_gamma


def expected_depth(n_remainder):
    """Calculates the expected isolation depth for a remainder of uniform points"""
    if n_remainder <= 1:
        return 0.0
    if n_remainder == 2:
        return 1.0
    return float(2 * (harmonic(n_remainder) - 1))


def calculate_depths(isolation_depths, tree, curr_node, curr_depth):
    """Fill in an array of isolation depths for a scikit-learn isolation forest model"""
    if tree.children_left[curr_node] == -1:
        isolation_depths[curr_node] = curr_depth + expected_depth(
            tree.n_node_samples[curr_node]
        )
    else:
        calculate_depths(
            isolation_depths, tree, tree.children_left[curr_node], curr_depth + 1
        )
        calculate_depths(
            isolation_depths, tree, tree.children_right[curr_node], curr_depth + 1
        )
