========
Treelite
========

**Treelite** is a universal model exchange and serialization format for
decision tree forests. Treelite aims to be a small library that enables
other C++ applications to exchange and store decision trees on the disk
as well as the network.

.. raw:: html

  <a class="github-button" href="https://github.com/dmlc/treelite"
     data-size="large" data-show-count="true"
     aria-label="Star dmlc/treelite on GitHub">Star</a>
  <a class="github-button" href="https://github.com/dmlc/treelite/subscription"
     data-size="large" data-show-count="true"
     aria-label="Watch dmlc/treelite on GitHub">Watch</a>

.. warning:: Tree compiler was migrated to TL2cgen

  If you are looking for a compiler to translate tree models into C code,
  use :doc:`TL2cgen <tl2cgen:index>`.
  To migrate existing code using Treelite 3.x, consult the page
  :doc:`tl2cgen:treelite-migration`.


Why Treelite?
=============

Universal, lightweight specification for all tree models
--------------------------------------------------------
Are you designing a C++ application that needs to read and write tree models,
e.g. a prediction server?
Do not be overwhelmed by the variety of tree models in the wild. Treelite
lets you convert many kinds of tree models into a **common specification**.
By using Treelite as a library, your application now only needs to deal
with one model specification instead of many. Treelite currently
supports:

* `XGBoost <https://github.com/dmlc/xgboost/>`_
* `LightGBM <https://github.com/Microsoft/LightGBM>`_
* `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_
* :doc:`flexible builder class <tutorials/builder>` for users of other
  tree libraries

In addition, tree libraries can directly output trained trees using the
Treelite specification. For example, the random forest algoritm in
`RAPIDS cuML <https://github.com/rapidsai/cuml>`_ stores the random forest
object using Treelite.

.. raw:: html

  <p>
  <a href="_static/deployment.png">
  <img src="_static/deployment.svg"
       onerror="this.src='_static/deployment.png'; this.onerror=null;"
       width="100%"><br>
  (Click to enlarge)
  </a>
  </p>

A small library that's easy to embed in another C++ application
---------------------------------------------------------------
Treelite has an up-to-date CMake build script. If your C++
application uses CMake, it is easy to embed Treelite.
Treelite is currently used by the following applications:

* :doc:`tl2cgen:index`
* Forest Inference Library (FIL) in `RAPIDS cuML <https://github.com/rapidsai/cuml>`_
* `Triton Inference Server FIL Backend <https://github.com/triton-inference-server/fil_backend>`_,
  an optimized prediction runtime for CPUs and GPUs.

Quick start
===========
Install Treelite:

.. code-block:: console

  # From PyPI
  pip install treelite
  # From Conda
  conda install -c conda-forge treelite

Import your tree ensemble model into Treelite:

.. code-block:: python

  import treelite
  model = treelite.frontend.load_xgboost_model("my_model.json")

Compute predictions using :doc:`treelite-gtil-api`:

.. code-block:: python

  X = ... # numpy array
  treelite.gtil.predict(model, data=X)

********
Contents
********

.. toctree::
  :maxdepth: 2
  :titlesonly:

  install
  tutorials/index
  treelite-api
  treelite-gtil-api
  treelite-c-api
  knobs/index
  serialization/index
  treelite-doxygen


*******
Indices
*******
* :ref:`genindex`
* :ref:`modindex`
