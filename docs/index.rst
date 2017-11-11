###############################################
Treelite : toolbox for decision tree deployment
###############################################

**Treelite** is a flexible toolbox for efficient deployment of decision tree
ensembles.

.. raw:: html

  <a class="github-button" href="https://github.com/dmlc/treelite"
     data-size="large" data-show-count="true"
     aria-label="Star dmlc/treelite on GitHub">Star</a>
  <a class="github-button" href="https://github.com/dmlc/treelite/subscription"
     data-size="large" data-show-count="true"
     aria-label="Watch dmlc/treelite on GitHub">Watch</a>

.. raw:: html

  <p>
  <a href="_static/deployment.png">
  <img src="_static/deployment.svg"
       onerror="this.src='_static/deployment.png'; this.onerror=null;"
       width="100%"><br>
  (Click to enlarge)
  </a>
  </p>

********
Features
********

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
the case no longer: treelite will export your model as a **stand-alone
prediction subroutine** so that predictions will be made without any machine
learning package installed.

Compile and optimize your model for fast prediction
===================================================
Treelite optimizes the prediction subroutine for faster prediction.

Depending on your use cases, simply compiling the prediction subroutine into
`machine code <https://en.wikipedia.org/wiki/Machine_code>`_ may boost the
performance noticeably. In addition, treelite supports additional optimizations
that improves performance while preserving the ensemble model.

Benchmark
^^^^^^^^^

The following figure shows the prediction throughput of treelite and XGBoost,
measured with various batch sizes.

.. plot:: _static/benchmark_plot.py
  :nofigs:

(`Plot code <_static/benchmark_plot.py>`_,
`SVG <_static/benchmark_plot.svg>`_, `PNG <_static/benchmark_plot.png>`_,
`High-resolution PNG <_static/benchmark_plot.hires.png>`_)

.. raw:: html

  <p>
  <img src="_static/benchmark_plot.svg"
       onerror="this.src='_static/benchmark_plot.png'; this.onerror=null;"
       width="100%">
  </p>

(Get exact measurements using the following links:
`allstate.csv <_static/allstate.csv>`_
`yahoo.csv <_static/yahoo.csv>`_)

**System configuration**. One AWS EC2 instance of type c4.8xlarge was used. It
consists of the following components:

* CPU: 36 virtual cores, 64-bit
* Memory: 60 GB
* Storage: Elastic Block Storage (EBS)
* Operating System: Ubuntu 14.04.5 LTS

**Datasets**. Two datasets were used.

* `Allstate Claim Prediction Challenge \
  <https://www.kaggle.com/c/ClaimPredictionChallenge>`_
* `Yahoo! Learning to Rank Challenge \
  <https://webscope.sandbox.yahoo.com/catalog.php?datatype=c>`_

**Methods**. For both datasets, we trained a 1600-tree ensemble using XGBoost.
Then we made predictions on batches of various sizes that were sampled randomly
from the training data. After running predictions using treelite and XGBoost
(latter with :py:meth:`xgboost.Booster.predict`), we measured throughput as
the number of lines predicted per second.

`Download the benchmark script <_static/benchmark.py>`_

**Caveats**. For datasets with a small number of features (< 30) and few missing
values, treelite may not produce any performance gain. The `higgs dataset
<https://archive.ics.uci.edu/ml/datasets/HIGGS>`_ is one such example:

.. plot:: _static/benchmark_plot2.py
  :nofigs:

(`Plot code <_static/benchmark_plot2.py>`_,
`SVG <_static/benchmark_plot2.svg>`_, `PNG <_static/benchmark_plot2.png>`_,
`High-resolution PNG <_static/benchmark_plot2.hires.png>`_)

.. raw:: html

  <p>
  <img src="_static/benchmark_plot2.svg"
       onerror="this.src='_static/benchmark_plot2.png'; this.onerror=null;"
       width="50%">
  </p>

We are investigating additional optimization strategies to further improve
performance.

********
Contents
********
The latest version of treelite is |version|.

.. toctree::
  :maxdepth: 2
  :titlesonly:

  install
  quick_start
  tutorials/index
  treelite-api
  treelite-runtime-api
  treelite-c-api
  Internal docs <http://treelite.readthedocs.io/en/latest/dev/>

*******
Indices
*******
* :ref:`genindex`
* :ref:`modindex`
