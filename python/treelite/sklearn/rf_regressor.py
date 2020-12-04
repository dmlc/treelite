# coding: utf-8
"""Converter for sklearn.ensemble.RandomForestRegressor"""
import treelite


class SKLRFRegressorMixin:
    """Mixin class to implement the converter for RandomForestRegressor"""

    @classmethod
    def process_model(cls, sklearn_model):
        """Process a RandomForestRegressor to convert it into a Treelite model"""
        # Initialize Treelite model builder
        # Set average_tree_output=True for random forests
        builder = treelite.ModelBuilder(
            num_feature=sklearn_model.n_features_, average_tree_output=True,
            threshold_type='float64', leaf_output_type='float64')

        # Iterate over individual trees
        for i in range(sklearn_model.n_estimators):
            # Process the i-th tree and add to the builder
            # process_tree() to be defined later
            builder.append(cls.process_tree(sklearn_model.estimators_[i].tree_,
                                            sklearn_model))

        return builder.commit()

    @classmethod
    def process_leaf_node(cls, treelite_tree, sklearn_tree, node_id, sklearn_model):
        # pylint: disable=W0613
        """Process a test node with a given node ID"""
        # The `value` attribute stores the output for every leaf node.
        leaf_value = sklearn_tree.value[node_id].squeeze()
        # Initialize the leaf node with given node ID
        treelite_tree[node_id].set_leaf_node(leaf_value, leaf_value_type='float64')
