import os

import treelite
import treelite_runtime
import numpy as np
import pytest
from treelite.contrib import _libext

from .util import os_compatible_toolchains

@pytest.fixture
def toy_model():
    builder = treelite.ModelBuilder(num_feature=1)
    tree = treelite.ModelBuilder.Tree()
    tree[0].set_categorical_test_node(feature_id=0, left_categories=[0], default_left=True,
                                      left_child_key=1, right_child_key=2)
    tree[1].set_leaf_node(-1.0)
    tree[2].set_leaf_node(1.0)
    tree[0].set_root()
    builder.append(tree)
    model = builder.commit()

    return model

@pytest.fixture
def test_data():
    return np.array([[-1], [-0.6], [-0.5], [0], [0.3], [0.7], [1]], dtype=np.float32)

@pytest.fixture
def ref_pred():
    # Negative inputs are mapped to the right child node
    # 0.3 and 0.7 are mapped to the left child node, since they get rounded toward the zero.
    return np.array([1, 1, 1, -1, -1, -1, 1], dtype=np.float32)

def test_gtil(toy_model, test_data, ref_pred):
    pred = treelite.gtil.predict(toy_model, test_data)
    np.testing.assert_equal(pred, ref_pred)

def test_treelite_compiled(tmpdir, toy_model, test_data, ref_pred):
    libpath = os.path.join(tmpdir, 'mylib' + _libext())
    toolchain = os_compatible_toolchains()[0]
    toy_model.export_lib(toolchain=toolchain, libpath=libpath)

    predictor = treelite_runtime.Predictor(libpath=libpath)
    dmat = treelite_runtime.DMatrix(test_data)
    pred = predictor.predict(dmat)
    np.testing.assert_equal(pred, ref_pred)
