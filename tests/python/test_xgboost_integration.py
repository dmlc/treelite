# -*- coding: utf-8 -*-
"""Tests for XGBoost integration"""
# pylint: disable=R0201, R0915
import json
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, lists, sampled_from

import treelite

from .util import TemporaryDirectory

try:
    import xgboost as xgb
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip("XGBoost not installed; skipping", allow_module_level=True)


@given(
    lists(integers(min_value=0, max_value=20), min_size=1, max_size=10),
    sampled_from(["string", "object", "list"]),
    sampled_from([True, False]),
)
@settings(print_blob=True, deadline=None)
def test_extra_field_in_xgb_json(random_integer_seq, extra_field_type, use_tempfile):
    # pylint: disable=too-many-locals,too-many-arguments
    """
    Test if we can handle extra fields in XGBoost JSON model file
    Insert an extra field at a random place and then load the model into Treelite,
    as follows:
    * Use the Hypothesis package to generate a random sequence of integers.
    * Then use the integer sequence to navigate through the XGBoost model JSON object.
    * Lastly, after navigating, insert the extra field at that location.
    """
    np.random.seed(0)
    nrow = 16
    ncol = 8
    X = np.random.randn(nrow, ncol)
    y = np.random.randint(0, 2, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == 1

    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "max_depth": 1,
        "eta": 1,
        "objective": "binary:logistic",
        "verbosity": 0,
    }
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=1,
    )
    model_json_str = bst.save_raw(raw_format="json").decode(encoding="utf-8")
    model_obj = json.loads(model_json_str)

    def get_extra_field_value():
        if extra_field_type == "object":
            return {}
        if extra_field_type == "string":
            return "extra"
        if extra_field_type == "list":
            return []
        return None

    def insert_extra_field(model_obj, seq):
        if (not seq) or (not model_obj):
            if isinstance(model_obj, dict):
                model_obj["extra_field"] = get_extra_field_value()
                return True
        elif isinstance(model_obj, list):
            idx = seq[0] % len(model_obj)
            subobj = model_obj[idx]
            if isinstance(subobj, (dict, list)):
                return insert_extra_field(subobj, seq[1:])
        elif isinstance(model_obj, dict):
            idx = seq[0] % len(model_obj)
            subobj = list(model_obj.items())[idx][1]
            if isinstance(subobj, (dict, list)):
                if not insert_extra_field(subobj, seq[1:]):
                    model_obj["extra_field"] = get_extra_field_value()
                return True
            model_obj["extra_field"] = get_extra_field_value()
            return True
        return False

    insert_extra_field(model_obj, random_integer_seq)
    new_model_str = json.dumps(model_obj)
    assert "extra_field" in new_model_str
    if use_tempfile:
        with TemporaryDirectory() as tmpdir:
            new_model_path = os.path.join(tmpdir, "new_model.json")
            with open(new_model_path, "w", encoding="utf-8") as f:
                f.write(new_model_str)
            treelite.Model.load(
                new_model_path, model_format="xgboost_json", allow_unknown_field=True
            )
    else:
        treelite.Model.from_xgboost_json(new_model_str, allow_unknown_field=True)
