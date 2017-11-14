=========
Benchmark
=========

The following figure shows the prediction throughput of treelite and XGBoost,
measured with various batch sizes.

.. plot:: _static/benchmark_plot.py
  :nofigs:

.. raw:: html

  <p>
  <img src="_static/benchmark_plot.svg"
       onerror="this.src='_static/benchmark_plot.png'; this.onerror=null;"
       width="100%">
  </p>

(Get this plot in `SVG <_static/benchmark_plot.svg>`_,
`PNG <_static/benchmark_plot.png>`_, 
`High-resolution PNG <_static/benchmark_plot.hires.png>`_)

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

**Actual measurements**. You may download the exact measurements using the
following links: `allstate.csv <_static/allstate.csv>`_
`yahoo.csv <_static/yahoo.csv>`_

**Caveats**. For datasets with a small number of features (< 30) and few missing
values, treelite may not produce any performance gain. The `higgs dataset
<https://archive.ics.uci.edu/ml/datasets/HIGGS>`_ is one such example:

.. plot:: _static/benchmark_plot2.py
  :nofigs:

.. raw:: html

  <p>
  <img src="_static/benchmark_plot2.svg"
       onerror="this.src='_static/benchmark_plot2.png'; this.onerror=null;"
       width="50%">
  </p>

(Get this plot in
`SVG <_static/benchmark_plot2.svg>`_, `PNG <_static/benchmark_plot2.png>`_,
`High-resolution PNG <_static/benchmark_plot2.hires.png>`_)

We are investigating additional optimization strategies to further improve
performance.