# process_tree(), process_node(), process_test_node() omitted to save space
# See the first section for their definitions

def process_model(sklearn_model):
  builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                  random_forest=True)
  for i in range(sklearn_model.n_estimators):
    # Process i-th tree and add to the builder
    builder.append( process_tree(sklearn_model.estimators_[i].tree_,
                                 sklearn_model) )

  return builder.commit()

def process_leaf_node(treelite_tree, sklearn_tree, node_id, sklearn_model):
  # Get counts for each label (+/-) at this leaf node
  leaf_count = sklearn_tree.value[node_id].squeeze()
  # Compute the fraction of positive data points at this leaf node
  fraction_positive = float(leaf_count[1]) / leaf_count.sum()
  # The fraction above is now the leaf output
  treelite_tree[node_id].set_leaf_node(fraction_positive)
