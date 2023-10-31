"""Treelite Model builder class"""
import ctypes
import dataclasses
import json
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

from .core import _LIB, _check_call
from .model import Model
from .util import c_array, c_str


@dataclasses.dataclass
class Metadata:
    """
    Metadata object, consisting of metadata information about the model at large.

    Parameters
    ----------
    num_feature:
        Number of features used in the model.
        We assume that all feature indices are between ``0`` and ``num_feature - 1``.
    task_type:
        Task type. Can be one of ``kBinaryClf``, ``kRegressor``, ``kMultiClf``,
        ``kLearningToRank``, or ``kIsolationForest``.
    average_tree_output:
        Whether to average outputs of trees
    num_target:
        Number of targets
    num_class:
        Number of classes. num_class[i] is the number of classes of target i.
    leaf_vector_shape:
        Shape of the output from each leaf node
    """

    num_feature: int
    task_type: str
    average_tree_output: bool
    num_target: int
    num_class: List[int]
    leaf_vector_shape: Tuple[int, int]

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TreeAnnotation:
    """
    Annotation for individual trees. Use this object to look up which target and class
    each tree is associated with.

    The output of each target / class is obtained by summing the outputs of all trees that are
    associated with that target / class.
    target_id[i] indicates the target the i-th tree is associated with.
    (-1 indicates that the tree is a multi-target tree, whose output gets counted for all targets.)
    class_id[i] indicates the class the i-th tree is associated with.
    (-1 indicates that the tree's output gets counted for all classes.)

    Parameters
    ----------
    num_tree:
        Number of trees
    target_id:
        Target that each tree is associated with (see explanation above)
    class_id:
        Class that each tree is associated with (see explanation above)
    """

    num_tree: int
    target_id: List[int]
    class_id: List[int]

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PostProcessorFunc:
    """
    Specification for postprocessor of prediction outputs

    Parameters
    ----------
    name:
        Name of the postprocessor
    sigmoid_alpha:
        Scaling parameter for sigmoid function ``sigmoid(x) = 1 / (1 + exp(-alpha * x))``.
        This parameter is applicable only when ``name="sigmoid"`` or ``name="multiclass_ova"``.
        It must be strictly positive.
    ratio_c:
        Scaling parameter for exponential standard ratio transformation
        ``expstdratio(x) = exp2(-x / c)``.
        This parameter is applicable only when ``name="exponential_standard_ratio"``.
    """

    name: str
    sigmoid_alpha: float = 1.0
    ratio_c: float = 1.0

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "config": {
                "sigmoid_alpha": self.sigmoid_alpha,
                "ratio_c": self.ratio_c,
            },
        }


class ModelBuilder:
    """
    Model builder class, to iteratively build a tree ensemble model.

    .. note::

        The model builder object must be only accessed by a single thread. To build
        multiple trees in parallel, create multiple builder objects and use model
        concatenation (:py:meth:`~treelite.Model.concatenate`).

    Parameters
    ----------
    threshold_type:
        Type of thresholds in the tree model
    leaf_output_type:
        Type of leaf outputs in the tree model
    metadata:
        Model metadata
    tree_annotation:
        Annotation for individual trees
    postprocessor:
        Postprocessor for prediction outputs
    base_scores:
        Baseline scores for targets and classes, before adding tree outputs.
        Also known as the intercept.
    attributes:
        Arbitrary JSON object, to be stored in the "attributes" field in the model object.
    """

    def __init__(
        self,
        *,
        threshold_type: str,
        leaf_output_type: str,
        metadata: Metadata,
        tree_annotation: TreeAnnotation,
        postprocessor: PostProcessorFunc,
        base_scores: List[float],
        attributes: Optional[Dict[Any, Any]] = None,
    ):  # pylint: disable=R0913
        self._handle = None

        handle = ctypes.c_void_p()
        metadata_obj = {
            "threshold_type": threshold_type,
            "leaf_output_type": leaf_output_type,
            "metadata": metadata.asdict(),
            "tree_annotation": tree_annotation.asdict(),
            "postprocessor": postprocessor.asdict(),
            "base_scores": base_scores,
        }
        if attributes is not None:
            metadata_obj["attributes"] = attributes
        metadata_str = json.dumps(metadata_obj)
        _check_call(
            _LIB.TreeliteGetModelBuilder(
                c_str(metadata_str),
                ctypes.byref(handle),
            )
        )
        self._handle = handle
        self.threshold_type = threshold_type
        self.leaf_output_type = leaf_output_type

    def __del__(self):
        if self.handle is not None:
            _check_call(_LIB.TreeliteDeleteModelBuilder(self._handle))
            self._handle = None

    @property
    def handle(self):
        """Access the handle to the associated C++ object"""
        return self._handle

    def start_tree(self):
        """Start a new tree"""
        _check_call(_LIB.TreeliteModelBuilderStartTree(self.handle))

    def end_tree(self):
        """End the current tree"""
        _check_call(_LIB.TreeliteModelBuilderEndTree(self.handle))

    def start_node(self, node_key: int):
        """
        Start a new node

        Parameters
        ----------
        node_key:
            Integer key that unique identifies the node
        """
        _check_call(
            _LIB.TreeliteModelBuilderStartNode(self.handle, ctypes.c_int(node_key))
        )

    def end_node(self):
        """End the current node"""
        _check_call(_LIB.TreeliteModelBuilderEndNode(self.handle))

    def numerical_test(
        self,
        feature_id: int,
        threshold: float,
        default_left: bool,
        opname: str,
        left_child_key: int,
        right_child_key: int,
    ):  # pylint: disable=R0913
        """
        Declare the current node as a numerical test node, where the test is of form
        [feature value] [op] [threshold]. Data points for which the test evaluates to True
        will be mapped to the left child node; all other data points (for which the test
        evaluates to False) will be mapped to the right child node.

        Parameters
        ----------
        feature_id:
            Feature ID
        threshold
            Threshold
        default_left
            Whether the missing value should be mapped to the left child
        opname
            Comparison operator
        left_child_key
            Integer key that unique identifies the left child node.
        right_child_key
            Integer key that unique identifies the right child node.
        """
        _check_call(
            _LIB.TreeliteModelBuilderNumericalTest(
                self.handle,
                ctypes.c_int32(feature_id),
                ctypes.c_double(threshold),
                ctypes.c_int(1 if default_left else 0),
                c_str(opname),
                ctypes.c_int(left_child_key),
                ctypes.c_int(right_child_key),
            )
        )

    def categorical_test(
        self,
        feature_id: int,
        default_left: bool,
        category_list: List[int],
        category_list_right_child: bool,
        left_child_key: int,
        right_child_key: int,
    ):  # pylint: disable=R0913
        """
        Declare the current node as a categorical test node, where the test is of form
        [feature value] \\in [category list].

        Parameters
        ----------
        feature_id:
            Feature ID
        default_left:
            Whether the missing value should be mapped to the left child
        category_list:
            List of categories to be tested for match
        category_list_right_child:
            Whether the data points for which the test evaluates to True should be mapped to the
            right child or the left child.
        left_child_key:
            Integer key that unique identifies the left child node.
        right_child_key:
            Integer key that unique identifies the right child node.
        """
        _check_call(
            _LIB.TreeliteModelBuilderCategoricalTest(
                self.handle,
                ctypes.c_int(feature_id),
                ctypes.c_bool(default_left),
                c_array(ctypes.c_uint32, category_list),
                ctypes.c_size_t(len(category_list)),
                ctypes.c_int(1 if category_list_right_child else 0),
                ctypes.c_int(left_child_key),
                ctypes.c_int(right_child_key),
            )
        )

    def leaf(self, leaf_value: Union[float, Sequence[float]]):
        """
        Declare the current node as a leaf node

        Parameters
        ----------
        leaf_value:
            Value of leaf output
        """
        if isinstance(leaf_value, Sequence):
            # leaf_value is a list
            if self.leaf_output_type == "float32":
                _check_call(
                    _LIB.TreeliteModelBuilderLeafVectorFloat32(
                        self.handle,
                        c_array(ctypes.c_float, leaf_value),
                        ctypes.c_size_t(len(leaf_value)),
                    )
                )
            elif self.leaf_output_type == "float64":
                _check_call(
                    _LIB.TreeliteModelBuilderLeafVectorFloat64(
                        self.handle,
                        c_array(ctypes.c_double, leaf_value),
                        ctypes.c_size_t(len(leaf_value)),
                    )
                )
        else:
            # leaf_value is a scalar
            _check_call(
                _LIB.TreeliteModelBuilderLeafScalar(
                    self.handle,
                    ctypes.c_double(leaf_value),
                )
            )

    def gain(self, gain: float):
        """
        Specify the gain (loss reduction) that's resulted from the current split.

        Parameters
        ----------
        gain:
            Gain (loss reduction)
        """
        _check_call(
            _LIB.TreeliteModelBuilderGain(
                self.handle,
                ctypes.c_double(gain),
            )
        )

    def data_count(self, data_count: int):
        """
        Specify the number of data points (samples) that are mapped to the current node.

        Parameters
        ----------
        data_count:
            Number of data points
        """
        _check_call(
            _LIB.TreeliteModelBuilderGain(
                self.handle,
                ctypes.c_uint64(data_count),
            )
        )

    def sum_hess(self, sum_hess: float):
        """
        Specify the weighted sample count or the sum of Hessians for the data points that
        are mapped to the current node.

        Parameters
        ----------
        sum_hess:
            Weighted sample count or the sum of Hessians
        """
        _check_call(
            _LIB.TreeliteModelBuilderSumHess(
                self.handle,
                ctypes.c_double(sum_hess),
            )
        )

    def commit(self) -> Model:
        """
        Conclude model building and obtain the final model object.

        Returns
        -------
        model : :py:class:`Model`
            Finished model
        """
        model_handle = ctypes.c_void_p()
        _check_call(
            _LIB.TreeliteModelBuilderCommitModel(
                self.handle,
                ctypes.byref(model_handle),
            )
        )
        return Model(handle=model_handle)
