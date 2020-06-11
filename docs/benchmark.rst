:orphan:

=========
Benchmark
=========

The following figure shows the prediction throughput of Treelite and XGBoost,
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

**System configuration**. One AWS EC2 instance of type c5.18xlarge was used. It
consists of the following components:

* CPU: 72 virtual cores, 64-bit
* Memory: 144 GB
* Storage: Elastic Block Storage (EBS)
* Operating System: Ubuntu 14.04.5 LTS

**Datasets**. Three datasets were used.

* `Allstate Claim Prediction Challenge \
  <https://www.kaggle.com/c/ClaimPredictionChallenge>`_
* `HIGGS Data Set \
  <https://archive.ics.uci.edu/ml/datasets/HIGGS>`_
* `Yahoo! Learning to Rank Challenge \
  <https://webscope.sandbox.yahoo.com/catalog.php?datatype=c>`_

**Methods**. For each datasets, we trained a 1600-tree ensemble using XGBoost.
Then we made predictions on batches of various sizes that were sampled randomly
from the training data. After running predictions using Treelite and XGBoost
(latter with :py:meth:`xgboost.Booster.predict`), we measured throughput as
the number of lines predicted per second.

Download the benchmark script: `benchmark.py <_static/benchmark.py>`_
`benchmark-xgb.py <_static/benchmark-xgb.py>`_

**Actual measurements**. You may download the exact measurements using the
following links:

* `allstate-treelite.csv <_static/allstate-treelite.csv>`_
* `allstate-xgb.csv <_static/allstate-xgb.csv>`_
* `higgs-treelite.csv <_static/higgs-treelite.csv>`_
* `higgs-xgb.csv <_static/higgs-xgb.csv>`_
* `yahoo-treelite.csv <_static/yahoo-treelite.csv>`_
* `yahoo-xgb.csv <_static/yahoo-xgb.csv>`_
