# coding: utf-8
"""Converter for sklearn.ensemble.GradientBoostingClassifier (multi-class classifier)"""
import treelite


class SKLGBMMultiClassifierMixin:
    """Mixin class to implement the converter for GradientBoostingClassifier
       (multi-class classifier)"""

    @classmethod
    def process_model(cls, sklearn_model):
        """Process a GradientBoostingClassifier (multi-class classifier) to convert it into a
           Treelite model"""
        # Check for init='zero'
        if sklearn_model.init != 'zero':
            raise treelite.TreeliteError("Gradient boosted trees must be trained with "
                                         "the option init='zero'")
        # Initialize Treelite model builder
        # Set average_tree_output=False for gradient boosted trees
        # Set num_class for multi-class classification
        # Set pred_transform='softmax' to obtain probability predictions
        builder = treelite.ModelBuilder(
            num_feature=sklearn_model.n_features_, num_class=sklearn_model.n_classes_,
            average_tree_output=False, pred_transform='softmax',
            threshold_type='float64', leaf_output_type='float64')
        # Process [number of iterations] * [number of classes] trees
        for i in range(sklearn_model.n_estimators):
            for k in range(sklearn_model.n_classes_):
                builder.append(cls.process_tree(sklearn_model.estimators_[i][k].tree_,
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
