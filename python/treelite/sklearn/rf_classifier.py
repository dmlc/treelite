# coding: utf-8
"""Converter for sklearn.ensemble.RandomForestClassifier (binary classifier)"""
import treelite


class SKLRFClassifierMixin:
    """Mixin class to implement the converter for RandomForestClassifier (binary classifier)"""

    @classmethod
    def process_model(cls, sklearn_model):
        """Process a RandomForestClassifier (binary classifier) to convert it into a
           Treelite model"""
        builder = treelite.ModelBuilder(
            num_feature=sklearn_model.n_features_, average_tree_output=True,
            threshold_type='float64', leaf_output_type='float64')
        for i in range(sklearn_model.n_estimators):
            # Process i-th tree and add to the builder
            builder.append(cls.process_tree(sklearn_model.estimators_[i].tree_,
                                            sklearn_model))

        return builder.commit()

    @classmethod
    def process_leaf_node(cls, treelite_tree, sklearn_tree, node_id, sklearn_model):
        # pylint: disable=W0613
        """Process a test node with a given node ID"""
        # Get counts for each label (+/-) at this leaf node
        leaf_count = sklearn_tree.value[node_id].squeeze()
        # Compute the fraction of positive data points at this leaf node
        fraction_positive = float(leaf_count[1]) / leaf_count.sum()
        # The fraction above is now the leaf output
        treelite_tree[node_id].set_leaf_node(fraction_positive, leaf_value_type='float64')
