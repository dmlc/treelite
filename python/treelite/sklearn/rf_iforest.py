# coding: utf-8
"""Converter for sklearn.ensemble.IsolationForest"""
from scipy.special import psi
from numpy import euler_gamma, zeros
import treelite


class SKLiForestMixin:
    """Mixin class to implement the converter for IsolationForest"""

    @classmethod
    def harmonic(cls, number):
        """Calculates the n-th harmonic number"""
        return psi(number+1) + euler_gamma

    @classmethod
    def expected_depth(cls, n_remainder): # pylint: disable=R1705
        """Calculates the expected isolation depth for a remainder of uniform points"""
        if n_remainder <= 1:
            return 0
        elif n_remainder == 2:
            return 1
        else:
            return float(2 * (cls.harmonic(n_remainder) - 1))


    @classmethod
    def calculate_depths(cls, isolation_depths, tree, curr_node, curr_depth):
        """Fill in an array of isolation depths for a scikit-learn isolation forest model"""
        if tree.children_left[curr_node] == -1:
            isolation_depths[curr_node] \
                = curr_depth + cls.expected_depth(tree.n_node_samples[curr_node])
        else:
            cls.calculate_depths(
                isolation_depths, tree, tree.children_left[curr_node], curr_depth+1)
            cls.calculate_depths(
                isolation_depths, tree, tree.children_right[curr_node], curr_depth+1)

    @classmethod
    def process_model(cls, sklearn_model):
        """Process a IsolationForest to convert it into a Treelite model"""
        # Initialize Treelite model builder
        # Set average_tree_output=True for random forests
        builder = treelite.ModelBuilder(
            num_feature=sklearn_model.n_features_, average_tree_output=True,
            threshold_type='float64', leaf_output_type='float64',
            pred_transform = 'exponential_standard_ratio',
            ratio_c = cls.expected_depth(sklearn_model.max_samples_))

        # Iterate over individual trees
        for i in range(sklearn_model.n_estimators):
            # First determine the actual isolation depths for the nodes
            isolation_depths = zeros(
                sklearn_model.estimators_[i].tree_.n_node_samples.shape[0],
                dtype = 'float64'
            )
            cls.calculate_depths(isolation_depths, sklearn_model.estimators_[i].tree_, 0, 0.0)

            # Process the i-th tree and add to the builder
            # process_tree() to be defined later
            builder.append(cls.process_tree(sklearn_model.estimators_[i].tree_,
                                            sklearn_model, isolation_depths))

        return builder.commit()

    @classmethod
    def process_leaf_node(cls, treelite_tree, sklearn_tree, node_id,
        sklearn_model, isolation_depths):  # pylint: disable=R0913
        # pylint: disable=W0613
        """Process a test node with a given node ID"""
        # The `value` attribute stores the output for every leaf node.
        leaf_value = isolation_depths[node_id]
        # Initialize the leaf node with given node ID
        treelite_tree[node_id].set_leaf_node(leaf_value, leaf_value_type='float64')
