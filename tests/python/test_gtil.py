"""
Tests for General Tree Inference Library (GTIL). The tests ensure that GTIL produces correct
prediction results for a variety of tree models.
"""
import os
import pytest
import treelite
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import scipy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, load_svmlight_file
from sklearn.model_selection import train_test_split

from .metadata import dataset_db
from .util import load_txt


@pytest.mark.parametrize('clazz', [RandomForestRegressor, ExtraTreesRegressor,
                                   GradientBoostingRegressor])
def test_skl_regressor(clazz):
    """Scikit-learn regressor"""
    X, y = load_boston(return_X_y=True)
    kwargs = {'max_depth': 3, 'random_state': 0, 'n_estimators': 10}
    if clazz == GradientBoostingRegressor:
        kwargs['init'] = 'zero'
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_pred = clf.predict(X)

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('clazz', [RandomForestClassifier, ExtraTreesClassifier,
                                   GradientBoostingClassifier])
def test_skl_binary_classifier(clazz):
    """Scikit-learn binary classifier"""
    X, y = load_breast_cancer(return_X_y=True)
    kwargs = {'max_depth': 3, 'random_state': 0, 'n_estimators': 10}
    if clazz == GradientBoostingClassifier:
        kwargs['init'] = 'zero'
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)[:, 1]

    tl_model = treelite.sklearn.import_model(clf)
    out_prob = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@pytest.mark.parametrize('clazz', [RandomForestClassifier, ExtraTreesClassifier,
                                   GradientBoostingClassifier])
def test_skl_multiclass_classifier(clazz):
    """Scikit-learn multi-class classifier"""
    X, y = load_iris(return_X_y=True)
    kwargs = {'max_depth': 3, 'random_state': 0, 'n_estimators': 10}
    if clazz == GradientBoostingClassifier:
        kwargs['init'] = 'zero'
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

    tl_model = treelite.sklearn.import_model(clf)
    out_prob = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@pytest.mark.parametrize('objective', ['reg:linear', 'reg:squarederror', 'reg:squaredlogerror',
                                       'reg:pseudohubererror'])
@pytest.mark.parametrize('model_format', ['binary', 'json'])
def test_xgb_boston(tmpdir, model_format, objective):
    # pylint: disable=too-many-locals
    """Test XGBoost with Boston data (regression)"""
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 8, 'eta': 1, 'silent': 1, 'objective': objective}
    num_round = 10
    xgb_model = xgb.train(param, dtrain, num_boost_round=num_round,
                          evals=[(dtrain, 'train'), (dtest, 'test')])
    if model_format == 'json':
        model_name = 'boston.json'
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)
        tl_model = treelite.Model.load(filename=model_path, model_format='xgboost_json')
    else:
        tl_model = treelite.Model.from_xgboost(xgb_model)

    out_pred = treelite.gtil.predict(tl_model, X_test)
    expected_pred = xgb_model.predict(dtest)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('model_format', ['binary', 'json'])
@pytest.mark.parametrize('objective', ['multi:softmax', 'multi:softprob'])
def test_xgb_iris(tmpdir, objective, model_format):
    # pylint: disable=too-many-locals
    """Test XGBoost with Iris data (multi-class classification)"""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 6, 'eta': 0.05, 'num_class': 3, 'verbosity': 0,
             'objective': objective, 'metric': 'mlogloss'}
    xgb_model = xgb.train(param, dtrain, num_boost_round=10,
                          evals=[(dtrain, 'train'), (dtest, 'test')])

    if model_format == 'json':
        model_name = 'iris.json'
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)
        tl_model = treelite.Model.load(filename=model_path, model_format='xgboost_json')
    else:
        tl_model = treelite.Model.from_xgboost(xgb_model)

    out_pred = treelite.gtil.predict(tl_model, X_test)
    expected_pred = xgb_model.predict(dtest)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('model_format', ['binary', 'json'])
@pytest.mark.parametrize('objective,max_label',
                         [('binary:logistic', 2),
                          ('binary:hinge', 2),
                          ('binary:logitraw', 2),
                          ('count:poisson', 4),
                          ('rank:pairwise', 5),
                          ('rank:ndcg', 5),
                          ('rank:map', 5)],
                         ids=['binary:logistic', 'binary:hinge', 'binary:logitraw',
                              'count:poisson', 'rank:pairwise', 'rank:ndcg', 'rank:map'])
def test_xgb_nonlinear_objective(tmpdir, objective, max_label, model_format):
    # pylint: disable=too-many-locals
    """Test XGBoost with non-linear objectives with dummy data"""
    nrow = 16
    ncol = 8
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(nrow, ncol), dtype=np.float32)
    y = rng.integers(0, max_label, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == max_label - 1

    num_round = 4
    dtrain = xgb.DMatrix(X, label=y)
    if objective.startswith('rank:'):
        dtrain.set_group([nrow])
    xgb_model = xgb.train({'objective': objective, 'base_score': 0.5, 'seed': 0},
                          dtrain=dtrain, num_boost_round=num_round)

    objective_tag = objective.replace(':', '_')
    if model_format == 'json':
        model_name = f'nonlinear_{objective_tag}.json'
    else:
        model_name = f'nonlinear_{objective_tag}.bin'
    model_path = os.path.join(tmpdir, model_name)
    xgb_model.save_model(model_path)

    tl_model = treelite.Model.load(
        filename=model_path,
        model_format=('xgboost_json' if model_format == 'json' else 'xgboost'))

    out_pred = treelite.gtil.predict(tl_model, X)
    expected_pred = xgb_model.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_xgb_categorical_split():
    """Test toy XGBoost model with categorical splits"""
    dataset = 'xgb_toy_categorical'
    tl_model = treelite.Model.load(dataset_db[dataset].model, model_format='xgboost_json')

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    expected_pred = load_txt(dataset_db[dataset].expected_margin)
    out_pred = treelite.gtil.predict(tl_model, X.toarray())
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('model_format', ['binary', 'json'])
def test_xgb_dart(tmpdir, model_format):
    # pylint: disable=too-many-locals
    """Test XGBoost DART model with dummy data"""
    nrow = 16
    ncol = 8
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(nrow, ncol))
    y = rng.integers(0, 2, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == 1

    num_round = 50
    dtrain = xgb.DMatrix(X, label=y)
    param = {'booster': 'dart',
             'max_depth': 5, 'learning_rate': 0.1,
             'objective': 'binary:logistic',
             'sample_type': 'uniform',
             'normalize_type': 'tree',
             'rate_drop': 0.1,
             'skip_drop': 0.5}
    xgb_model = xgb.train(param, dtrain=dtrain, num_boost_round=num_round)

    if model_format == 'json':
        model_name = 'dart.json'
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)
        tl_model = treelite.Model.load(filename=model_path, model_format='xgboost_json')
    else:
        tl_model = treelite.Model.from_xgboost(xgb_model)
    out_pred = treelite.gtil.predict(tl_model, X)
    expected_pred = xgb_model.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('objective', ['regression', 'regression_l1', 'huber'])
@pytest.mark.parametrize('reg_sqrt', [True, False])
def test_lightgbm_regression(tmpdir, objective, reg_sqrt):
    # pylint: disable=too-many-locals
    """Test LightGBM regressor"""
    lgb_model_path = os.path.join(tmpdir, 'boston_lightgbm.txt')

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {'task': 'train', 'boosting_type': 'gbdt', 'objective': objective, 'reg_sqrt': reg_sqrt,
             'metric': 'rmse', 'num_leaves': 31, 'learning_rate': 0.05}
    lgb_model = lgb.train(param, dtrain, num_boost_round=10, valid_sets=[dtrain, dtest],
                          valid_names=['train', 'test'])
    expected_pred = lgb_model.predict(X_test)
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.Model.load(lgb_model_path, model_format='lightgbm')
    out_pred = treelite.gtil.predict(tl_model, X_test)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('objective', ['binary', 'xentlambda', 'xentropy'])
def test_lightgbm_binary_classification(tmpdir, objective):
    # pylint: disable=too-many-locals
    """Test LightGBM binary classifier"""
    lgb_model_path = os.path.join(tmpdir, 'breast_cancer_lightgbm.txt')
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {'task': 'train', 'boosting_type': 'gbdt', 'objective': objective, 'metric': 'auc',
             'num_leaves': 7, 'learning_rate': 0.1}
    lgb_model = lgb.train(param, dtrain, num_boost_round=10, valid_sets=[dtrain, dtest],
                               valid_names=['train', 'test'])
    expected_prob = lgb_model.predict(X_test)
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.Model.load(lgb_model_path, model_format='lightgbm')
    out_prob = treelite.gtil.predict(tl_model, X_test)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@pytest.mark.parametrize('boosting_type', ['gbdt', 'rf'])
@pytest.mark.parametrize('objective', ['multiclass', 'multiclassova'])
def test_lightgbm_multiclass_classification(tmpdir, objective, boosting_type):
    # pylint: disable=too-many-locals
    """Test LightGBM multi-class classifier"""
    lgb_model_path = os.path.join(tmpdir, 'iris_lightgbm.txt')
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {'task': 'train', 'boosting': boosting_type, 'objective': objective,
             'metric': 'multi_logloss', 'num_class': 3, 'num_leaves': 31, 'learning_rate': 0.05}
    if boosting_type == 'rf':
        param.update({'bagging_fraction': 0.8, 'bagging_freq': 1})
    lgb_model = lgb.train(param, dtrain, num_boost_round=10, valid_sets=[dtrain, dtest],
                               valid_names=['train', 'test'])
    expected_pred = lgb_model.predict(X_test)
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.Model.load(lgb_model_path, model_format='lightgbm')
    out_pred = treelite.gtil.predict(tl_model, X_test)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_lightgbm_categorical_data():
    """Test LightGBM with toy categorical data"""
    dataset = 'toy_categorical'
    lgb_model_path = dataset_db[dataset].model
    tl_model = treelite.Model.load(lgb_model_path, model_format='lightgbm')

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    expected_pred = load_txt(dataset_db[dataset].expected_margin)
    out_pred = treelite.gtil.predict(tl_model, X.toarray())
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_lightgbm_sparse_ranking_model(tmpdir):
    """Generate a LightGBM ranking model with highly sparse data."""
    rng = np.random.default_rng(seed=2020)
    X = scipy.sparse.random(m=10, n=206947, format='csr', dtype=np.float64, random_state=0,
                            density=0.0001)
    X.data = rng.standard_normal(size=X.data.shape[0], dtype=np.float64)
    y = rng.integers(low=0, high=5, size=X.shape[0])

    params = {
        'objective': 'lambdarank',
        'num_leaves': 32,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'min_gain_to_split': 0.0,
        'learning_rate': 1.0,
        'min_data_in_leaf': 1
    }

    lgb_model_path = os.path.join(tmpdir, 'sparse_ranking_lightgbm.txt')

    dtrain = lgb.Dataset(X, label=y, group=[X.shape[0]])
    lgb_model = lgb.train(params, dtrain, num_boost_round=1)
    lgb_out = lgb_model.predict(X)
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.Model.load(lgb_model_path, model_format='lightgbm')

    # GTIL doesn't yet support sparse matrix; so use NaN to represent missing values
    Xa = X.toarray()
    Xa[Xa == 0] = 'nan'
    out = treelite.gtil.predict(tl_model, Xa)

    np.testing.assert_almost_equal(out, lgb_out)


def test_lightgbm_sparse_categorical_model():
    """Test LightGBM with high-cardinality categorical features"""
    dataset = 'sparse_categorical'
    lgb_model_path = dataset_db[dataset].model
    tl_model = treelite.Model.load(lgb_model_path, model_format='lightgbm')

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True,
                              n_features=tl_model.num_feature)
    expected_pred = load_txt(dataset_db[dataset].expected_margin)
    out_pred = treelite.gtil.predict(tl_model, X.toarray(), pred_margin=True)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)
