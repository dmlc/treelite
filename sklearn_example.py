from treelite import DMatrix
from treelite.compiler import Compiler
from treelite.frontend import ModelBuilder
from treelite.contrib import create_shared
from treelite.predictor import Predictor

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys

# load a multi-class classification problem
X, y = load_wine(return_X_y=True)
ntree = 5

# 1. random forest
clf = RandomForestClassifier(random_state=1, n_estimators=ntree)
clf.fit(X, y)

builder = ModelBuilder(num_feature=clf.n_features_,
                       num_output_group=clf.n_classes_,
                       params={'pred_transform':'identity_multiclass'})
                       # no need for softmax here, as each leaf node already
                       # produces a probability distribution

for i in range(ntree):
  clf_tree = clf.estimators_[i].tree_
  nnode = clf_tree.node_count
  node_value = clf_tree.value.squeeze()

  tree = ModelBuilder.Tree()
  tree[0].set_root()
  for nid in range(nnode):
    if clf_tree.children_left[nid] == -1:  # leaf
      # scikit-learn random forests remember the distribution of classes
      # among the training instances in each leaf node.
      tree[nid].set_leaf_node(node_value[nid] / node_value[nid].sum())
    else:  # non-leaf
      tree[nid].set_numerical_test_node(
                                  feature_id=clf_tree.feature[nid],
                                  opname='<=',
                                  threshold=clf_tree.threshold[nid],
                                  default_left=True,
                                  left_child_key=clf_tree.children_left[nid],
                                  right_child_key=clf_tree.children_right[nid])
  builder.append(tree)
model = builder.commit()
compiler = Compiler()
compiler.compile(model, dirpath='test_rf', params={}, verbose=True)
create_shared('gcc', 'test_rf', verbose=True)
predictor = Predictor('test_rf', verbose=True)
dmat = DMatrix(X, verbose=True)
out_pred = predictor.predict(dmat, verbose=True)

# 2. gradient boosting
clf = GradientBoostingClassifier(random_state=1, n_estimators=ntree,
                                 max_depth=4, init='zero')
                           # set init='zero' to start boosting from zero.
                           # scikit-learn by default starts boosting from
                           # the prior distribution of classes, but tree-lite
                           # does not yet support boosting from nonzero.
clf.fit(X, y)

builder = ModelBuilder(num_feature=clf.n_features_,
                       num_output_group=clf.n_classes_,
                       params={'pred_transform':'softmax'})

for i in range(ntree):
  for k in range(clf.n_classes_):
    clf_tree = clf.estimators_[i][k].tree_
    nnode = clf_tree.node_count
    node_value = clf_tree.value.squeeze()

    tree = ModelBuilder.Tree()
    tree[0].set_root()
    for nid in range(nnode):
      if clf_tree.children_left[nid] == -1:  # leaf
        tree[nid].set_leaf_node(clf.learning_rate * node_value[nid])
            # for scikit-learn gradient booster, each leaf value is "unscaled";
            # we'd need to shrink it by the learning rate.
      else:  # non-leaf
        tree[nid].set_numerical_test_node(
                                    feature_id=clf_tree.feature[nid],
                                    opname='<=',
                                    threshold=clf_tree.threshold[nid],
                                    default_left=True,
                                    left_child_key=clf_tree.children_left[nid],
                                    right_child_key=clf_tree.children_right[nid])
    builder.append(tree)
model = builder.commit()
compiler.compile(model, dirpath='test_gbm', params={'parallel_comp':5}, verbose=True)
create_shared('gcc', 'test_gbm', verbose=True)
predictor = Predictor('test_gbm', verbose=True)
dmat = DMatrix(X, verbose=True)
out_pred = predictor.predict(dmat, verbose=True)
