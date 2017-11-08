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
  <img src="_static/deployment.svg"
       onerror="this.src='_static/deployment.png'; this.onerror=null;"
       width="100%">
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

**[Insert benchmark here]**

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
