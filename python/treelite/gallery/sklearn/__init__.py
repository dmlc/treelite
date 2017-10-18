# coding: utf-8

from __future__ import absolute_import as _abs
import os
import treelite

def import_model(sklearn_model):
  """
  Load a tree ensemble model from a scikit-learn model object

  Parameters
  ----------
  sklearn_model : object of type \
                  :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                  :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                  :py:class:`~sklearn.ensemble.GradientBoostingRegressor` / \
                  :py:class:`~sklearn.ensemble.GradientBoostingClassifier`
      Python handle to scikit-learn model

  Returns
  -------
  model : :py:class:`~treelite.Model` object
      loaded model

  Example
  -------

  .. code-block:: python
    :emphasize-lines: 8

    import sklearn.datasets
    import sklearn.ensemble
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    clf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
    clf.fit(X, y)

    import treelite.gallery.sklearn
    model = treelite.gallery.sklearn.import_model(clf)
  """
  class_name = sklearn_model.__class__.__name__
  module_name = sklearn_model.__module__.split('.')[0]

  if module_name != 'sklearn':
    raise Exception('Not a scikit-learn model')

  _execfile('common.py')

  if class_name == 'RandomForestRegressor':
    _execfile('rf_regressor.py')
  elif class_name == 'RandomForestClassifier':
    if sklearn_model.n_classes_ == 2:
      _execfile('rf_classifier.py')
    elif sklearn_model.n_classes_ > 2:
      _execfile('rf_multi_classifier.py')
    else:
      raise Exception('n_classes_ must be at least 2')
  elif class_name == 'GradientBoostingRegressor':
    _execfile('gbm_regressor.py')
  elif class_name == 'GradientBoostingClassifier':
    if sklearn_model.n_classes_ == 2:
      _execfile('gbm_classifier.py')
    elif sklearn_model.n_classes_ > 2:
      _execfile('gbm_multi_classifier.py')
    else:
      raise Exception('n_classes_ must be at least 2')
  else:
    raise Exception('Unsupported model type: only '
                    'random forests and gradient boosted trees are supported')

  return process_model(sklearn_model)

def _execfile(filename):
  fullpath = os.path.join(os.path.dirname(__file__), filename)
  with open(fullpath) as f:
    code = compile(f.read(), filename, 'exec')
    exec(code, globals())

__all__ = ['import_model']
