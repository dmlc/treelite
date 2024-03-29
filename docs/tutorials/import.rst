Importing tree ensemble models
==============================

Since the scope of Treelite is limited to **prediction** only, one must use
other machine learning packages to **train** decision tree ensemble models. In
this document, we will show how to import an ensemble model that had been
trained elsewhere.

.. contents:: Contents
  :local:

Importing XGBoost models
------------------------

**XGBoost** (`dmlc/xgboost <https://github.com/dmlc/xgboost/>`_) is a fast,
scalable package for gradient boosting. Both Treelite and XGBoost are hosted
by the DMLC (Distributed Machine Learning Community) group.

Treelite plays well with XGBoost --- if you used XGBoost to train your ensemble
model, you need only one line of code to import it. Depending on where your
model is located, use :py:meth:`~treelite.frontend.from_xgboost`,
:py:meth:`~treelite.frontend.load_xgboost_model`, or
:py:meth:`~treelite.frontend.load_xgboost_model_legacy_binary`:

* Load XGBoost model from a :py:class:`xgboost.Booster` object

.. code-block:: python

  # bst = an object of type xgboost.Booster
  model = treelite.frontend.from_xgboost(bst)

* Load XGBoost model from a model file

.. code-block:: python

  # JSON format
  model = treelite.frontend.load_xgboost_model("my_model.json")
  # Legacy binary format
  model = treelite.frontend.load_xgboost_model_legacy_binary("my_model.model")

Importing LightGBM models
-------------------------

**LightGBM** (`Microsoft/LightGBM <https://github.com/Microsoft/LightGBM>`_) is
another well known machine learning package for gradient boosting. To import
models generated by LightGBM, use the
:py:meth:`~treelite.frontend.load_lightgbm_model` method:

.. code-block:: python

  model = treelite.frontend.load_lightgbm_model("lightgbm_model.txt")

Importing scikit-learn models
-----------------------------
**Scikit-learn** (`scikit-learn/scikit-learn
<https://github.com/scikit-learn/scikit-learn>`_) is a Python machine learning
package known for its versatility and ease of use. It supports a wide variety
of models and algorithms. The following kinds of models can be imported into
Treelite.

* :py:class:`sklearn.ensemble.RandomForestRegressor`
* :py:class:`sklearn.ensemble.RandomForestClassifier`
* :py:class:`sklearn.ensemble.ExtraTreesRegressor`
* :py:class:`sklearn.ensemble.ExtraTreesClassifier`
* :py:class:`sklearn.ensemble.GradientBoostingRegressor`
* :py:class:`sklearn.ensemble.GradientBoostingClassifier`
* :py:class:`sklearn.ensemble.HistGradientBoostingRegressor`
* :py:class:`sklearn.ensemble.HistGradientBoostingClassifier`
* :py:class:`sklearn.ensemble.IsolationForest`

To import scikit-learn models, use
:py:meth:`treelite.sklearn.import_model`:

.. code-block:: python

  # clf is the model object generated by scikit-learn
  import treelite.sklearn
  model = treelite.sklearn.import_model(clf)

How about other packages?
-------------------------
If you used other packages to train your ensemble model, you'd need to specify
the model programmatically:

* :doc:`/tutorials/builder`
