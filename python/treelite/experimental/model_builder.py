"""
Next-gen model builder, using dataclasses.

Limitations
* Only one kind of comparison op used throughout the model
* float32 type only
"""

import ctypes
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Union

import numpy as np

from ..core import _LIB, _check_call, c_array
from ..frontend import Model
from ..sklearn.importer import ArrayOfArrays
from ..util import c_str


@dataclass
class TaskParam:
    output_type: Literal["float", "int"]
    grove_per_class: bool
    num_class: int
    leaf_vector_size: int


@dataclass
class ModelParam:
    pred_transform: str
    sigmoid_alpha: float = 1.0
    ratio_c: float = 1.0
    global_bias: float = 0.0


@dataclass
class NumericalSplitNode:
    split_feature_id: int
    default_left: bool
    left_child: int
    right_child: int
    threshold: float
    split_type: Literal["numerical"] = "numerical"


@dataclass
class CategoricalSplitNode:
    split_feature_id: int
    default_left: bool
    left_child: int
    right_child: int
    categories_list_right_child: bool
    categories_list: List[int]
    split_type: Literal["categorical"] = "categorical"


@dataclass
class LeafNode:
    leaf_value: Union[float, List[float]]


@dataclass
class Tree:
    nodes: Dict[int, Union[NumericalSplitNode, CategoricalSplitNode, LeafNode]]


@dataclass
class ModelSpec:
    num_feature: int
    average_tree_output: bool
    comparison_op: Literal["==", "<", "<=", ">", ">="]
    task_type: Literal[
        "BinaryClfRegr",
        "MultiClfGrovePerClass",
        "MultiClfProbDistLeaf",
        "MultiClfCategLeaf",
    ]
    task_param: TaskParam
    model_param: ModelParam
    trees: List[Tree]

    def commit(self) -> Model:
        """Convert to Treelite model"""
        # Marshal data into arrays and then invoke TreeliteBuildModelFromArrays
        handle = ctypes.c_void_p()

        metadata = {
            "num_tree": len(self.trees),
            "num_feature": self.num_feature,
            "average_tree_output": self.average_tree_output,
            "comparison_op": self.comparison_op,
            "task_type": self.task_type,
            "task_param": asdict(self.task_param),
            "model_param": asdict(self.model_param),
        }
        metadata = json.dumps(metadata)

        node_count = []
        split_type = ArrayOfArrays(dtype=np.int8)
        default_left = ArrayOfArrays(dtype=np.int8)
        children_left = ArrayOfArrays(dtype=np.int32)
        children_right = ArrayOfArrays(dtype=np.int32)
        split_feature = ArrayOfArrays(dtype=np.uint32)
        threshold = ArrayOfArrays(dtype=np.float32)
        leaf_value = ArrayOfArrays(dtype=np.float32)
        categories_list = ArrayOfArrays(dtype=np.uint32)
        categories_list_offset_begin = ArrayOfArrays(dtype=np.int64)
        categories_list_offset_end = ArrayOfArrays(dtype=np.int64)
        categories_list_right_child = ArrayOfArrays(dtype=np.int8)

        for tree in self.trees:
            max_node_id = max(tree.nodes)
            node_count.append(max_node_id + 1)

            split_type_l = []
            default_left_l = []
            children_left_l = []
            children_right_l = []
            split_feature_l = []
            threshold_l = []
            leaf_value_l = []
            categories_list_l = []
            categories_list_offset_begin_l = []
            categories_list_offset_end_l = []
            categories_list_right_child_l = []
            for node_id in range(max_node_id + 1):
                if node_id not in tree.nodes:
                    # gap in node ID; fill it with empty node
                    split_type_l.append(0)
                    default_left_l.append(False)
                    children_left_l.append(-1)
                    children_right_l.append(-1)
                    split_feature_l.append(-1)
                    threshold_l.append(np.nan)
                    leaf_value_l.append(np.nan)
                    categories_list_offset_begin_l.append(len(categories_list_l))
                    categories_list_offset_end_l.append(len(categories_list_l))
                    categories_list_right_child_l.append(False)
                    continue
                node = tree.nodes[node_id]
                if isinstance(node, LeafNode):
                    split_type_l.append(0)
                    default_left_l.append(False)
                    children_left_l.append(-1)
                    children_right_l.append(-1)
                    split_feature_l.append(-1)
                    threshold_l.append(np.nan)
                    if isinstance(node.leaf_value, list):
                        if len(node.leaf_value) != self.task_param.leaf_vector_size:
                            raise ValueError(
                                f"Leaf output must have length "
                                f"{self.task_param.leaf_vector_size}"
                            )
                        leaf_value_l.extend(node.leaf_value)
                    else:
                        if self.task_param.leaf_vector_size != 1:
                            raise ValueError(
                                f"Leaf output must have length "
                                f"{self.task_param.leaf_vector_size}"
                            )
                        leaf_value_l.append(node.leaf_value)
                    categories_list_offset_begin_l.append(len(categories_list_l))
                    categories_list_offset_end_l.append(len(categories_list_l))
                    categories_list_right_child_l.append(False)
                elif isinstance(node, NumericalSplitNode):
                    split_type_l.append(1)
                    default_left_l.append(node.default_left)
                    children_left_l.append(node.left_child)
                    children_right_l.append(node.right_child)
                    split_feature_l.append(node.split_feature_id)
                    threshold_l.append(node.threshold)
                    leaf_value_l.append(np.nan)
                    categories_list_offset_begin_l.append(len(categories_list_l))
                    categories_list_offset_end_l.append(len(categories_list_l))
                    categories_list_right_child_l.append(False)
                elif isinstance(node, CategoricalSplitNode):
                    split_type_l.append(2)
                    default_left_l.append(node.default_left)
                    children_left_l.append(node.left_child)
                    children_right_l.append(node.right_child)
                    split_feature_l.append(node.split_feature_id)
                    threshold_l.append(np.nan)
                    leaf_value_l.append(np.nan)
                    categories_list_offset_begin_l.append(len(categories_list_l))
                    categories_list_offset_end_l.append(
                        len(categories_list_l) + len(node.categories_list)
                    )
                    categories_list_l.extend(node.categories_list)
                    categories_list_right_child_l.append(
                        node.categories_list_right_child
                    )
                else:
                    raise ValueError(f"Unknown node type: {node.__class__.__name__}")
            split_type.add(split_type_l)
            default_left.add(default_left_l)
            children_left.add(children_left_l)
            children_right.add(children_right_l)
            split_feature.add(split_type_l)
            threshold.add(threshold_l)
            leaf_value.add(leaf_value_l)
            categories_list.add(categories_list_l)
            categories_list_offset_begin.add(categories_list_offset_begin_l)
            categories_list_offset_end.add(categories_list_offset_end_l)
            categories_list_right_child.add(categories_list_right_child_l)

        _check_call(
            _LIB.TreeliteBuildModelFromArrays(
                c_str(metadata),
                c_array(ctypes.c_int64, node_count),
                split_type.as_c_array(),
                default_left.as_c_array(),
                children_left.as_c_array(),
                children_right.as_c_array(),
                split_feature.as_c_array(),
                threshold.as_c_array(),
                leaf_value.as_c_array(),
                categories_list.as_c_array(),
                categories_list_offset_begin.as_c_array(),
                categories_list_offset_end.as_c_array(),
                categories_list_right_child.as_c_array(),
                ctypes.byref(handle),
            )
        )
        return Model(handle)
