# coding: utf-8
"""Common facilities for all scikit-learn converters"""
import treelite


class SKLConverterBase:
    """Define methods common to all converters"""

    @classmethod
    def process_tree(cls, sklearn_tree, sklearn_model):
        """Process a scikit-learn Tree object"""
        treelite_tree = treelite.ModelBuilder.Tree(
            threshold_type='float64', leaf_output_type='float64')

        # Iterate over each node: node ID ranges from 0 to [node_count]-1
        for node_id in range(sklearn_tree.node_count):
            cls.process_node(treelite_tree, sklearn_tree, node_id, sklearn_model)

        # Node #0 is always root for scikit-learn decision trees
        treelite_tree[0].set_root()

        return treelite_tree

    @classmethod
    def process_node(cls, treelite_tree, sklearn_tree, node_id, sklearn_model):
        """Process a tree node in a scikit-learn Tree object. Decide whether the node is
           a leaf node or a test node."""
        if sklearn_tree.children_left[node_id] == -1:  # leaf node
            cls.process_leaf_node(treelite_tree, sklearn_tree, node_id, sklearn_model)
        else:  # test node
            cls.process_test_node(treelite_tree, sklearn_tree, node_id, sklearn_model)

    @classmethod
    def process_test_node(cls, treelite_tree, sklearn_tree, node_id, sklearn_model):
        # pylint: disable=W0613
        """Process a test node with a given node ID. We shall assume that all tree ensembles in
           scikit-learn use only numerical splits."""
        treelite_tree[node_id].set_numerical_test_node(
            feature_id=sklearn_tree.feature[node_id],
            opname='<=',
            threshold=sklearn_tree.threshold[node_id],
            threshold_type='float64',
            default_left=True,
            left_child_key=sklearn_tree.children_left[node_id],
            right_child_key=sklearn_tree.children_right[node_id],)

    @classmethod
    def process_leaf_node(cls, treelite_tree, sklearn_tree, node_id, sklearn_model):
        """Process a test node with a given node ID. This method shall be implemented by the
           mixin class that represents the converter for a particular model type."""
        raise NotImplementedError()
