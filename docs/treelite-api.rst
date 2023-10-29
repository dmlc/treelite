============
Treelite API
============

API of Treelite Python package.

.. contents::
  :local:

Model loaders
-------------

.. automodule:: treelite.frontend
   :members:
   :member-order: bysource

Scikit-learn importer
---------------------

.. automodule:: treelite.sklearn
   :members:
   :member-order: bysource

Model builder
-------------

.. automodule:: treelite.model_builder
    :members:
    :member-order: bysource

Model builder (Legacy)
----------------------

.. autoclass:: treelite.ModelBuilder
    :members:
    :member-order: bysource

Model class
-----------

.. autoclass:: treelite.Model
   :members:
   :member-order: bysource


.. _field_accessors:

Field accessors (Advanced)
--------------------------
Using field accessors, users can query and modify the value of fields in a :py:class:`~treelite.Model` object.

.. note:: Modifying a field is an unsafe operation

  Treelite does not prevent users from assigning an invalid value to a field. Setting an invalid value may
  cause undefined behavior. Always consult :doc:`the model spec </serialization/v4>` to carefully examine
  model invariants and constraints on fields. For example, most tree fields must have an array of length ``num_nodes``.

.. code-block:: python

  import treelite
  import numpy as np

  # model is treelite.Model object

  # Get the "num_feature" field in the header
  model.get_header_accessor().get_field("num_feature")
  # Modify the "postprocessor" field in the header. Use length-1 array to indicate scalar
  new_value = np.array([100], dtype=np.int32)
  model.get_header_accessor().set_field("num_feature", new_value)

  # Get the "postprocessor" field in the header
  model.get_header_accessor().get_field("postprocessor")
  # Modify the "postprocessor" field in the header
  model.get_header_accessor().set_field("postprocessor", "identity")

  # Get the "leaf_value" field in the first tree
  print(model.get_tree_accessor(0).get_field("leaf_value"))
  # Modify the "leaf_value" field in the first tree
  new_value = np.array([0, 0, 0.5, 1, -0.5], dtype=np.float32)
  model.get_tree_accessor(0).set_field("leaf_value", new_value)

.. autoclass:: treelite.model.HeaderAccessor
   :members:

.. autoclass:: treelite.model.TreeAccessor
   :members:
