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
See :doc:`/tutorials/edit` for more details.

.. note:: Modifying a field is an unsafe operation

  Treelite does not prevent users from assigning an invalid value to a field. Setting an invalid value may
  cause undefined behavior. Always consult :doc:`the model spec </serialization/v4>` to carefully examine
  model invariants and constraints on fields. For example, most tree fields must have an array of length ``num_nodes``.

.. autoclass:: treelite.model.HeaderAccessor
   :members:

.. autoclass:: treelite.model.TreeAccessor
   :members:
