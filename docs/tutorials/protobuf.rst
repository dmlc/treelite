Specifying models using Protocol Buffers
========================================
Since the scope of Treelite is limited to **prediction** only, one must use
other machine learning packages to **train** decision tree ensemble models. In
this document, we will show how to import an ensemble model that had been
trained elsewhere.

**Using XGBoost or LightGBM for training?** Read :doc:`this document <import>`
instead.

.. contents:: Contents
  :local:
  :backlinks: none

What is Protocol Buffers?
-------------------------
**Protocol Buffers** (`google/protobuf <https://github.com/google/protobuf>`_)
is a widely used mechanism to serialize structured data. You may specify your
ensemble model according to the specification `src/tree.proto
<https://github.com/dmlc/treelite/blob/master/src/tree.proto>`_. Depending on
the package you used to train the model, it may take some effort to express
the model in terms of the given spec. See `this helpful guide
<https://developers.google.com/protocol-buffers/docs/tutorials>`_ on reading
and writing serialized messages.

To import models that had been serialized with Protocol Buffers,
use the :py:meth:`~treelite.Model.load` method with argument
``format='protobuf'``:

.. code-block:: python

  # model had been saved to a file named my_model.bin
  # notice the second argument format='protobuf'
  model = Model.load('my_model.bin', format='protobuf')
