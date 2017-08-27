from treelite import Compiler, ModelBuilder

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys

# load a multi-class classification problem
X, y = load_wine(return_X_y=True)
ntree = 30

# 1. random forest
clf = RandomForestClassifier(random_state=1, n_estimators=ntree)
clf.fit(X, y)

builder = ModelBuilder(num_feature=clf.n_features_,
                       num_output_group=clf.n_classes_)

for i in range(ntree):
  clf_tree = clf.estimators_[i].tree_
  nnode = clf_tree.node_count
  node_value = clf_tree.value.squeeze()

  tree = ModelBuilder.Tree()
  tree[0].set_root()
  for nid in range(nnode):
    if clf_tree.children_left[nid] == -1:  # leaf
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
compiler = Compiler("recursive")
compiler.generate_code(model, 'test_rf', params={}, verbose=True)

# 2. gradient boosting
clf = GradientBoostingClassifier(random_state=1, n_estimators=ntree, max_depth=4)
clf.fit(X, y)

builder = ModelBuilder(num_feature=clf.n_features_,
                       num_output_group=clf.n_classes_)

for i in range(ntree):
  for k in range(clf.n_classes_):    
    clf_tree = clf.estimators_[i][k].tree_
    nnode = clf_tree.node_count
    node_value = clf_tree.value.squeeze()

    tree = ModelBuilder.Tree()
    tree[0].set_root()
    for nid in range(nnode):
      if clf_tree.children_left[nid] == -1:  # leaf
        tree[nid].set_leaf_node(node_value[nid])
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
compiler.generate_code(model, 'test_gbm', params={}, verbose=True)