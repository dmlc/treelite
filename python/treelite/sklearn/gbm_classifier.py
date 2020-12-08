# coding: utf-8
"""Converter for sklearn.ensemble.GradientBoostingClassifier (binary classifier)"""
import treelite


class SKLGBMClassifierMixin:
    """Mixin class to implement the converter for GradientBoostingClassifier (binary classifier)"""

    @classmethod
    def process_model(cls, sklearn_model):
        """Process a GradientBoostingClassifier (binary classifier) to convert it into a
           Treelite model"""
        # Check for init='zero'
        if sklearn_model.init != 'zero':
            raise treelite.TreeliteError("Gradient boosted trees must be trained with "
                                         "the option init='zero'")
        # Initialize Treelite model builder
        # Set average_tree_output=False for gradient boosted trees
        # Set pred_transform='sigmoid' to obtain probability predictions
        builder = treelite.ModelBuilder(
            num_feature=sklearn_model.n_features_, average_tree_output=False,
            pred_transform='sigmoid', threshold_type='float64', leaf_output_type='float64')
        for i in range(sklearn_model.n_estimators):
            # Process i-th tree and add to the builder
            builder.append(cls.process_tree(sklearn_model.estimators_[i][0].tree_,
                                            sklearn_model))

        return builder.commit()

    @classmethod
    def process_leaf_node(cls, treelite_tree, sklearn_tree, node_id, sklearn_model):
        """Process a test node with a given node ID"""
        leaf_value = sklearn_tree.value[node_id].squeeze()
        # Need to shrink each leaf output by the learning rate
        leaf_value *= sklearn_model.learning_rate
        # Initialize the leaf node with given node ID
        treelite_tree[node_id].set_leaf_node(leaf_value, leaf_value_type='float64')
