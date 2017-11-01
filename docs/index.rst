###############################################
Treelite : toolbox for decision tree deployment
###############################################

**Treelite** is a flexible toolbox for efficient deployment of decision tree
ensembles.

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
`scikit-learn <https://github.com/scikit-learn/scikit-learn>`_. In cases you
are using another package to train your model, you may use the
`flexible builder class <tutorials/builder.html>`_.

Deploy only the parts you need
==============================
It is a great hassle to install machine learning packages (e.g. XGBoost,
LightGBM, scikit-learn, etc.) on every machine your tree model will run. This is
the case no longer: treelite will "compile" your model into a small prediction
library so that **predictions will be made without any machine learning package
installed.**

Depending on the machine you'd like to deploy your model, you may choose
three levels of dependencies.

.. raw:: html

  <p>
  <img src="_static/dependencies.svg"
       onerror="this.src='_static/dependencies.png'; this.onerror=null;"
       width="100%">
  </p>

* Option 1: Install treelite on the target machine
* Option 2: Deploy prediction code with a small runtime (helper code)
* Option 3: Deploy prediciton code only

Option 1 is the most convenient option but requires the highest number of
dependencies. Option 3, on the other hand, is the least convenient and requires
the least number of dependencies. In fast, it requires only a working C compiler
on the target machine. Option 2 is a compromise between the other two options,
in which only a **small runtime** portion of treelite (consisting of helper
functions) is deployed to the target machine. Follow the links above for more
details.

Compile and optimize your model for fast prediction
===================================================
Given any decision tree ensemble model, treelite will produce
**executable code** that makes predictions using that model. Treelite is able to
make compile-time optimizations to make prediction more efficient.

Here are some optimizations in place. Click each link for more information.

* Embed model information into machine instructions, by translating trees into
  sequences of if-then-else blocks
* Replace floating-point thresholds with integer indices
* Use the prediction paths of training data points to annotate branches

We are interested in adding more optimizations in the future.

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
