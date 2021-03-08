#####################################################
Treelite : model compiler for decision tree ensembles
#####################################################

**Treelite** is a model compiler for decision tree ensembles, aimed at
efficient deployment.

.. raw:: html

  <a class="github-button" href="https://github.com/dmlc/treelite"
     data-size="large" data-show-count="true"
     aria-label="Star dmlc/treelite on GitHub">Star</a>
  <a class="github-button" href="https://github.com/dmlc/treelite/subscription"
     data-size="large" data-show-count="true"
     aria-label="Watch dmlc/treelite on GitHub">Watch</a>

|longrelease|

*************
Why Treelite?
*************

Compile and optimize your model for fast prediction
===================================================
**Treelite compiles your tree model into optimized shared library.**
A :py:doc:`benchmark` demonstrates 2-6x improvement in prediction throughput,
due to more efficient use of compute resources.

.. raw:: html

  <p>
  <img src="_static/benchmark_plot.svg"
       onerror="this.src='_static/benchmark_plot.png'; this.onerror=null;"
       width="100%">
  </p>

Use machine learning package of your choice
===========================================
Treelite accommodates a wide range of decision tree ensemble models. In
particular, it handles both
`random forests <https://en.wikipedia.org/wiki/Random_forest>`_ and
`gradient boosted trees <https://en.wikipedia.org/wiki/Gradient_boosting>`_.

Treelite can read models produced by
`XGBoost <https://github.com/dmlc/xgboost/>`_,
`LightGBM <https://github.com/Microsoft/LightGBM>`_, and
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`_. In cases where
you are using another package to train your model, you may use the
:doc:`flexible builder class <tutorials/builder>`.

Deploy with minimal dependencies
================================
It is a great hassle to install machine learning packages (e.g. XGBoost,
LightGBM, scikit-learn, etc.) on every machine your tree model will run. This is
the case no longer: Treelite will export your model as a stand-alone
prediction library so that predictions will be made without any machine
learning package installed.

***********
Quick start
***********
Install Treelite from PyPI:

.. code-block:: console

  python3 -m pip install --user treelite treelite_runtime

Import your tree ensemble model into Treelite:

.. code-block:: python

  import treelite
  model = treelite.Model.load('my_model.model', model_format='xgboost')

Deploy a source archive:

.. code-block:: python

  # Produce a zipped source directory, containing all model information
  # Run `make` on the target machine
  model.export_srcpkg(platform='unix', toolchain='gcc',
                      pkgpath='./mymodel.zip', libname='mymodel.so',
                      verbose=True)

Deploy a shared library:

.. code-block:: python

  # Like export_srcpkg, but generates a shared library immediately
  # Use this only when the host and target machines are compatible
  model.export_lib(toolchain='gcc', libpath='./mymodel.so', verbose=True)

Make predictions on the target machine:

.. code-block:: python

  import treelite_runtime
  predictor = treelite_runtime.Predictor('./mymodel.so', verbose=True)
  dmat = treelite_runtime.DMatrix(X)
  out_pred = predictor.predict(dmat)

Read :doc:`tutorials/first` for a more detailed example. See
:doc:`tutorials/deploy` for additional instructions on deployment.

.. note:: A note on API compatibility

  Since Treelite is in early development, its API may change substantially
  in the future.

*********
Benchmark
*********

.. raw:: html

  <p>
  <img src="_static/benchmark_plot.svg"
       onerror="this.src='_static/benchmark_plot.png'; this.onerror=null;"
       width="100%">
  </p>

See the page :py:doc:`benchmark` for details.

******************
How Treelite works
******************

.. raw:: html

  <p>
  <a href="_static/deployment.png">
  <img src="_static/deployment.svg"
       onerror="this.src='_static/deployment.png'; this.onerror=null;"
       width="100%"><br>
  (Click to enlarge)
  </a>
  </p>

The workflow involves two distinct machines: **the host machine** that generates
prediction subroutine from a given tree model, and **the target machine** that
runs the subroutine. The two machines exchange a single C file that contains
all relevant information about the tree model. Only the host machine needs to
have Treelite installed; the target machine requires only a working C compiler.

********
Contents
********

.. toctree::
  :maxdepth: 2
  :titlesonly:

  install
  tutorials/index
  treelite-api
  treelite-runtime-api
  treelite-c-api
  javadoc/packages
  Treelite runtime Rust API <http://dovahcrow.github.io/treerite/treerite/>
  knobs/index
  Internal docs <http://treelite.readthedocs.io/en/latest/dev/>
  

*******
Indices
*******
* :ref:`genindex`
* :ref:`modindex`
