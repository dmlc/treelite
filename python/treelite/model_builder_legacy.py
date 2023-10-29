"""Treelite Model builder class (legacy)"""
# pylint: disable=E1135, E1137

from __future__ import annotations

import dataclasses
import warnings
from typing import Dict, List, Optional, Union

from .core import TreeliteError
from .model import Model
from .model_builder import Metadata
from .model_builder import ModelBuilder as ModelBuilderNew
from .model_builder import PostProcessorFunc, TreeAnnotation


@dataclasses.dataclass
class _LeafNode:
    leaf_value: Union[float, List[float]]


@dataclasses.dataclass
class _NumericalTestNode:
    feature_id: int
    opname: str
    threshold: float
    default_left: bool
    left_child_key: int
    right_child_key: int


@dataclasses.dataclass
class _CategoricalTestNode:
    feature_id: int
    left_categories: List[int]
    default_left: bool
    left_child_key: int
    right_child_key: int


class ModelBuilder:
    """
    Legacy model builder class. **New code should use the new model builder**
    :py:class:`treelite.model_builder.ModelBuilder` **instead.**

    This module is meant to enable existing code using the old model builder API
    to continue functioning. Users are highly encouraged to migrate to the new
    model builder API, to take advantage of new features including the support
    for multi-target tree models.

    Parameters
    ----------
    num_feature :
        Number of features used in model being built. We assume that all
        feature indices are between ``0`` and ``num_feature - 1``.
    num_class :
        Number of output groups; ``>1`` indicates multiclass classification
    average_tree_output :
        Whether the model is a random forest; ``True`` indicates a random forest
        and ``False`` indicates gradient boosted trees
    threshold_type:
        Type of thresholds in the tree model
    leaf_output_type:
        Type of leaf outputs in the tree model
    pred_transform:
        Postprocessor for prediction outputs
    sigmoid_alpha:
        Scaling parameter for sigmoid function ``sigmoid(x) = 1 / (1 + exp(-alpha * x))``.
        This parameter is applicable only when ``name="sigmoid"`` or ``name="multiclass_ova"``.
        It must be strictly positive.
    ratio_c:
        Scaling parameter for exponential standard ratio transformation
        ``expstdratio(x) = exp2(-x / c)``.
        This parameter is applicable only when ``name="exponential_standard_ratio"``.
    global_bias:
        Global bias of the model.
        Predicted margin scores of all instances will be  adjusted by the global bias.
    """

    class Node:
        """A node in a tree"""

        def __init__(self):
            self.empty = True
            self.node_key: Optional[int] = None
            self.tree: Optional[ModelBuilder.Tree] = None
            self.node: Optional[
                Union[_LeafNode, _NumericalTestNode, _CategoricalTestNode]
            ] = None

        def __repr__(self):
            return "<treelite.ModelBuilder.Node object>"

        def set_root(self):
            """Set the node as the root"""
            if self.tree:
                self.tree._set_root(node_key=self.node_key)  # pylint: disable=W0212
            else:
                raise TreeliteError(
                    "This node has never been inserted into a tree; "
                    "a node must be inserted before it can be a root"
                )

        def set_leaf_node(
            self,
            leaf_value: Union[float, List[float]],
            leaf_value_type: str = "float32",
        ):
            """
            Set the node as a leaf node

            Parameters
            ----------
            leaf_value :
                Usually a single leaf value (weight) of the leaf node. For multi-class
                random forest classifier, leaf_value should be a list of leaf weights.
            leaf_value_type :
                Data type used for leaf_value
            """

            if not self.empty:
                raise ValueError(
                    "Cannot modify a non-empty node. "
                    "If you meant to change type of the node, "
                    "delete it first and then add an empty node with "
                    "the same key."
                )
            if self.tree is None:
                raise TreeliteError(
                    "This node has never been inserted into a tree; "
                    "a node must be inserted before it can be a leaf node"
                )
            if self.tree.leaf_output_type != leaf_value_type:
                raise ValueError(
                    "Leaf value must have same type as the tree's leaf_output_type "
                    f"({self.tree.leaf_output_type})."
                )
            self.node = _LeafNode(leaf_value=leaf_value)

        # pylint: disable=R0913
        def set_numerical_test_node(
            self,
            feature_id: int,
            opname: str,
            threshold: float,
            default_left: bool,
            left_child_key: int,
            right_child_key: int,
            threshold_type: str = "float32",
        ):
            """
            Set the node as a test node with numerical split. The test is in the form
            ``[feature value] OP [threshold]``. Depending on the result of the test,
            either left or right child would be taken.

            Parameters
            ----------
            feature_id :
                Feature index
            opname :
                Binary operator to use in the test
            threshold :
                Threshold value
            default_left :
                Default direction for missing values
                (``True`` for left; ``False`` for right)
            left_child_key :
                Unique integer key to identify the left child node
            right_child_key :
                Unique integer key to identify the right child node
            threshold_type :
                Data type for threshold value (e.g. 'float32')
            """
            if not self.empty:
                raise ValueError(
                    "Cannot modify a non-empty node. "
                    "If you meant to change type of the node, "
                    "delete it first and then add an empty node with "
                    "the same key."
                )
            if self.tree is None:
                raise TreeliteError(
                    "This node has never been inserted into a tree; "
                    "a node must be inserted before it can be a test node"
                )
            # Automatically create child nodes that don't exist yet
            if left_child_key not in self.tree:
                self.tree[left_child_key] = ModelBuilder.Node()
            if right_child_key not in self.tree:
                self.tree[right_child_key] = ModelBuilder.Node()
            if self.tree.threshold_type != threshold_type:
                raise ValueError(
                    "Threshold value must have same type as the tree's threshold_type "
                    f"({self.tree.threshold_type})."
                )
            self.node = _NumericalTestNode(
                feature_id=feature_id,
                opname=opname,
                threshold=threshold,
                default_left=default_left,
                left_child_key=left_child_key,
                right_child_key=right_child_key,
            )
            self.empty = False

        # pylint: disable=R0913
        def set_categorical_test_node(
            self,
            feature_id: int,
            left_categories: List[int],
            default_left: bool,
            left_child_key: int,
            right_child_key: int,
        ):
            """
            Set the node as a test node with categorical split. A list defines all
            categories that would be classified as the left side. Categories are
            integers ranging from ``0`` to ``n-1``, where ``n`` is the number of
            categories in that particular feature.

            Parameters
            ----------
            feature_id :
                Feature index
            left_categories :
                List of categories belonging to the left child.
            default_left :
                Default direction for missing values
                (``True`` for left; ``False`` for right)
            left_child_key :
                Unique integer key to identify the left child node
            right_child_key :
                Unique integer key to identify the right child node
            """
            if not self.empty:
                raise ValueError(
                    "Cannot modify a non-empty node. "
                    "If you meant to change type of the node, "
                    "delete it first and then add an empty node with "
                    "the same key."
                )
            if self.tree is None:
                raise TreeliteError(
                    "This node has never been inserted into a tree; "
                    "a node must be inserted before it can be a test node"
                )
            # Automatically create child nodes that don't exist yet
            if left_child_key not in self.tree:
                self.tree[left_child_key] = ModelBuilder.Node()
            if right_child_key not in self.tree:
                self.tree[right_child_key] = ModelBuilder.Node()
            self.node = _CategoricalTestNode(
                feature_id=feature_id,
                left_categories=left_categories,
                default_left=default_left,
                left_child_key=left_child_key,
                right_child_key=right_child_key,
            )
            self.empty = False

    class Tree:
        """
        A decision tree in a tree ensemble builder

        Parameters
        ----------
        threshold_type:
            Type of thresholds in the tree model
        leaf_output_type:
            Type of leaf outputs in the tree model
        """

        def __init__(
            self,
            threshold_type: str = "float32",
            leaf_output_type: str = "float32",
        ):
            self.ensemble: Optional[ModelBuilder] = None
            self.nodes: Dict[int, ModelBuilder.Node] = {}
            self.threshold_type = threshold_type
            self.leaf_output_type = leaf_output_type

        # Implement dict semantics whenever applicable
        def items(self):  # pylint: disable=C0111
            return self.nodes.items()

        def keys(self):  # pylint: disable=C0111
            return self.nodes.keys()

        def values(self):  # pylint: disable=C0111
            return self.nodes.values()

        def __len__(self):
            return len(self.nodes)

        def __getitem__(self, key):
            if key not in self.nodes:
                # Implicitly create a new node
                self.__setitem__(key, ModelBuilder.Node())
            return self.nodes.__getitem__(key)

        def __setitem__(self, key, value):
            if not isinstance(value, ModelBuilder.Node):
                raise ValueError("Value must be of type ModelBuidler.Node")
            if key in self.nodes:
                raise KeyError(
                    "Nodes with duplicate keys are not allowed. "
                    f"If you meant to replace node {key}, "
                    "delete it first and then add an empty node with "
                    "the same key."
                )
            if not value.empty:
                raise ValueError("Can only insert an empty node")
            self.nodes.__setitem__(key, value)
            value.node_key = key  # Save node id for later
            value.tree = self

        def __delitem__(self, key):
            self.nodes.__delitem__(key)

        def __iter__(self):
            return self.nodes.__iter__()

        def __repr__(self):
            return f"<treelite.ModelBuilder.Tree object containing {len(self.nodes)} nodes>\n"

        def __contains__(self, item):
            return self.nodes.__contains__(item)

        def _set_root(self, node_key: Optional[int]):
            assert node_key is not None
            self.root_key = node_key

    def __init__(
        self,
        num_feature: int,
        num_class: int = 1,
        average_tree_output: bool = False,
        threshold_type: str = "float32",
        leaf_output_type: str = "float32",
        *,
        pred_transform: str = "identity",
        sigmoid_alpha: float = 1.0,
        ratio_c: float = 1.0,
        global_bias: float = 0.0,
    ):  # pylint: disable=R0913
        warnings.warn(
            "treelite.ModelBuilder is deprecated and will be removed in Treelite 4.1. "
            "Please use treelite.model_builder.ModelBuilder class instead. The new model builder "
            "supports many more features such as support for multi-target tree models.",
            FutureWarning,
        )
        if num_class == 1:
            if pred_transform == "sigmoid":
                task_type = "kBinaryClf"
            else:
                task_type = "kRegressor"
            grove_per_class = False
        elif num_class > 1:
            task_type = "kMultiClf"
            grove_per_class = True
        else:
            raise ValueError("num_class must be at least 1")
        if pred_transform == "max_index":
            # max_index not supported; replace with softmax
            pred_transform = "softmax"
        if threshold_type == "unit32" or leaf_output_type == "unit32":
            raise ValueError(
                "Integer type (uint32) is no longer supported."
                "Please use float32 or float64 for thresholds and leaf outputs."
            )
        self.trees: List[ModelBuilder.Tree] = []
        self.threshold_type = threshold_type
        self.leaf_output_type = leaf_output_type
        self.metadata = Metadata(
            num_feature=num_feature,
            task_type=task_type,
            average_tree_output=average_tree_output,
            num_target=1,
            num_class=[num_class],
            leaf_vector_shape=(1, 1),
        )
        self.grove_per_class = grove_per_class
        self.postprocessor = PostProcessorFunc(
            name=pred_transform,
            sigmoid_alpha=sigmoid_alpha,
            ratio_c=ratio_c,
        )
        self.base_scores = [global_bias]

    def insert(self, index: int, tree: ModelBuilder.Tree):
        """
        Insert a tree at specified location in the ensemble

        Parameters
        ----------
        index :
            Index of the element before which to insert the tree
        tree :
            Tree to be inserted
        """
        if not isinstance(index, int):
            raise ValueError("index must be of int type")
        if index < 0 or index > len(self):
            raise ValueError("index out of bounds")
        if not isinstance(tree, ModelBuilder.Tree):
            raise ValueError("tree must be of type ModelBuilder.Tree")
        tree.ensemble = self
        self.trees.insert(index, tree)

    def append(self, tree: ModelBuilder.Tree):
        """
        Add a tree at the end of the ensemble

        Parameters
        ----------
        tree :
            tree to be added
        """
        if not isinstance(tree, ModelBuilder.Tree):
            raise ValueError("tree must be of type ModelBuilder.Tree")
        self.insert(len(self), tree)

    def commit(self) -> Model:
        """
        Finalize the ensemble model

        Returns
        -------
        model : :py:class:`Model`
            Finished model
        """
        num_tree = len(self.trees)
        if self.grove_per_class:
            tree_annotation = TreeAnnotation(
                num_tree=len(self.trees),
                target_id=[0] * num_tree,
                class_id=[(i % self.metadata.num_class[0]) for i in range(num_tree)],
            )
        else:
            tree_annotation = TreeAnnotation(
                num_tree=len(self.trees),
                target_id=[0] * num_tree,
                class_id=[0] * num_tree,
            )
        builder = ModelBuilderNew(
            threshold_type=self.threshold_type,
            leaf_output_type=self.leaf_output_type,
            metadata=self.metadata,
            tree_annotation=tree_annotation,
            postprocessor=self.postprocessor,
            base_scores=self.base_scores,
        )
        for tree in self.trees:
            builder.start_tree()
            root_node = tree[tree.root_key].node
            del tree.nodes[tree.root_key]
            nodes = [(tree.root_key, root_node)] + [
                (k, v.node) for k, v in tree.nodes.items()
            ]
            for key, node in nodes:
                builder.start_node(key)
                if isinstance(node, _LeafNode):
                    # Leaf root
                    builder.leaf(node.leaf_value)
                elif isinstance(node, _NumericalTestNode):
                    builder.numerical_test(
                        feature_id=node.feature_id,
                        threshold=node.threshold,
                        default_left=node.default_left,
                        opname=node.opname,
                        left_child_key=node.left_child_key,
                        right_child_key=node.right_child_key,
                    )
                elif isinstance(node, _CategoricalTestNode):
                    builder.categorical_test(
                        feature_id=node.feature_id,
                        default_left=node.default_left,
                        category_list=node.left_categories,
                        category_list_right_child=False,
                        left_child_key=node.left_child_key,
                        right_child_key=node.right_child_key,
                    )
                else:
                    raise ValueError("Root node cannot be empty")
                builder.end_node()
            builder.end_tree()
        return builder.commit()

    # Implement list semantics whenever applicable
    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        return self.trees.__getitem__(index)

    def __delitem__(self, index):
        self.trees.__delitem__(index)

    def __iter__(self):
        return self.trees.__iter__()

    def __reversed__(self):
        return self.trees.__reversed__()

    def __repr__(self):
        return (
            f"<treelite.ModelBuilder object storing {len(self.trees)} decision trees>\n"
        )
