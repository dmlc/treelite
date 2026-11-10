import numpy as np
import pytest

import treelite

try:
    import xgboost as xgb
    from xgboost.testing.data import make_categorical
except ImportError:
    pytest.skip("XGBoost not installed; skipping", allow_module_level=True)


try:
    import pandas as pd
except ImportError:
    pytest.skip("Pandas required", allow_module_level=True)


def test_categorical_info():
    n_samples = int(2**10)
    X, y = make_categorical(
        n_samples,
        n_features=2,
        n_categories=3,
        onehot=False,
        cat_ratio=1.0,
        cat_dtype=np.str_,
        device="cpu",
    )
    X2, y2 = make_categorical(
        n_samples,
        n_features=3,
        n_categories=3,
        onehot=False,
        cat_ratio=2 / 3,
        cat_dtype=int,
        device="cpu",
    )
    X2 = X2.rename(columns={"0": "2", "1": "3", "2": "4"})
    Xc = pd.concat([X, X2], axis=1)
    yc = y + y2

    dmat = xgb.DMatrix(Xc, label=yc, enable_categorical=True)
    params = {"max_depth": 1, "objective": "reg:squarederror"}
    bst = xgb.train(params, dmat, num_boost_round=1, evals=[(dmat, "train")])

    treelite_model = treelite.frontend.from_xgboost(bst)
    cat_info = treelite_model.categorical_info

    # Features 0-1: Categorical features with string categories
    # Features 2-3: Categorical features with integer categories
    # Feature 4: Numerical feature
    for feature_id in [0, 1, 2, 3]:
        assert feature_id in cat_info
    assert 4 not in cat_info

    for feature_id in [0, 1, 2, 3]:
        cat_vals = cat_info[feature_id]["categories"]
        assert cat_vals == Xc[str(feature_id)].cat.categories.tolist()
        assert cat_info[feature_id]["sorted_idx"] == np.argsort(cat_vals).tolist()
