Editing tree models (Advanced)
==============================

:ref:`The field accessor API <field_accessors>` allows users to inspect and edit tree model objects after they have been
constructed. Here are some examples:

.. code-block:: python

  import treelite
  import numpy as np

  # model is treelite.Model object

  # Get the "num_feature" field in the header
  model.get_header_accessor().get_field("num_feature")
  # Modify the "num_feature" field in the header. Use length-1 array to indicate scalar
  new_value = np.array([100], dtype=np.int32)
  model.get_header_accessor().set_field("num_feature", new_value)

  # Get the "postprocessor" field in the header
  model.get_header_accessor().get_field("postprocessor")
  # Modify the "postprocessor" field in the header
  model.get_header_accessor().set_field("postprocessor", "identity")

  # Get the "leaf_value" field in the first tree
  model.get_tree_accessor(0).get_field("leaf_value")
  # Modify the "leaf_value" field in the first tree
  new_value = np.array([0, 0, 0.5, 1, -0.5], dtype=np.float32)
  model.get_tree_accessor(0).set_field("leaf_value", new_value)

Consult :doc:`/serialization/v4` for the list of fields.

How to use setter methods
-------------------------
There are lots of gotchas and pitfalls involved when using :py:meth:`~treelite.model.TreeAccessor.set_field` to modify
trees. We must start by the following notice:

.. note:: Modifying a field is an unsafe operation

  Treelite does not prevent users from assigning an invalid value to a field. Setting an invalid value may
  cause undefined behavior. Always consult :doc:`the model spec </serialization/v4>` to carefully examine
  model invariants and constraints on fields.

Make sure to follow the rules below to prevent errors and silent crashes:

* Always pass in a NumPy array when calling :py:meth:`~treelite.model.TreeAccessor.set_field`, even when the field
  being set is a scalar.
* Make sure to use the correct ``dtype`` when passing in the NumPy array. For example, since ``num_feature`` has
  type ``int32_t`` according to the model spec, use ``np.array([...], dtype=np.int32)``.
* Most of the fields accessed through the tree accessor :py:class:`~treelite.model.TreeAccessor` must have their values
  set to arrays of length ``num_nodes``, where ``num_nodes`` is the number of nodes. Setting a shorter array will
  likely cause undefined behavior and silent crashes.
* When adding additional nodes, make sure to update the ``num_nodes`` field as well as all tree fields. Example:

  .. code-block:: python

    import treelite
    from treelite.model_builder import (
      Metadata,
      ModelBuilder,
      PostProcessorFunc,
      TreeAnnotation,
    )

    # Tree stump with 3 nodes
    builder = ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=Metadata(
            num_feature=2,
            task_type="kRegressor",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(num_tree=1, target_id=[0], class_id=[0]),
        postprocessor=PostProcessorFunc(name="identity"),
        base_scores=[0.0],
    )
    builder.start_tree()
    builder.start_node(0)
    builder.numerical_test(
        feature_id=0,
        threshold=0.0,
        default_left=False,
        opname="<=",
        left_child_key=1,
        right_child_key=2,
    )
    builder.end_node()
    builder.start_node(1)
    builder.leaf(-1.0)
    builder.end_node()
    builder.start_node(2)
    builder.leaf(1.0)
    builder.end_node()
    builder.end_tree()

    model = builder.commit()
    tree = model.get_tree_accessor(0)

    # Add a test node. The tree now has 5 nodes total
    tree.set_field("num_nodes", np.array([5], dtype=np.int32))
    tree.set_field("node_type", np.array([1, 0, 1, 0, 0], dtype=np.int8))
    tree.set_field("cleft", np.array([1, -1, 3, -1, -1], dtype=np.int32))
    tree.set_field("cright", np.array([2, -1, 4, -1, -1], dtype=np.int32))
    tree.set_field("split_index", np.array([0, -1, 1, -1, 1], dtype=np.int32))
    tree.set_field("default_left", np.array([0, 0, 0, 0, 0], dtype=np.int8))
    tree.set_field("leaf_value", np.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float32))
    tree.set_field("threshold", np.array([1.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32))
    tree.set_field("cmp", np.array([2, 0, 2, 0, 0], dtype=np.int8))
    tree.set_field("category_list_right_child", np.array([0] * 5, dtype=np.uint8))
    tree.set_field("leaf_vector_begin", np.array([0] * 5, dtype=np.uint64))
    tree.set_field("leaf_vector_end", np.array([0] * 5, dtype=np.uint64))
    tree.set_field("category_list_begin", np.array([0] * 5, dtype=np.uint64))
    tree.set_field("category_list_end", np.array([0] * 5, dtype=np.uint64))

* But really, if possible, avoid changing the number of nodes in the tree to avoid possible errors. Actions that don't
  change the tree structure, such as re-numbering feature IDs and changing leaf outputs, are much safer.

Currently, it is not possible to add or remove trees using the field accessor API.
