# coding: utf-8
"""Converter for sklearn.ensemble.RandomForestClassifier (multi-class classifier)"""
import treelite


class SKLRFMultiClassifierMixin:
    """Mixin class to implement the converter for RandomForestClassifier (multi-class classifier)"""

    @classmethod
    def process_model(cls, sklearn_model):
        """Process a RandomForestClassifier (multi-class classifier) to convert it into a
           Treelite model"""
        # Must specify num_class and pred_transform
        builder = treelite.ModelBuilder(
            num_feature=sklearn_model.n_features_, num_class=sklearn_model.n_classes_,
            average_tree_output=True, pred_transform='identity_multiclass',
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
        # Get counts for each label class at this leaf node
        leaf_count = sklearn_tree.value[node_id].squeeze()
        # Compute the probability distribution over label classes
        prob_distribution = leaf_count / leaf_count.sum()
        # The leaf output is the probability distribution
        treelite_tree[node_id].set_leaf_node(prob_distribution, leaf_value_type='float64')
