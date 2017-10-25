def process_model(sklearn_model):
  # Initialize treelite model builder
  # Set random_forest=True for random forests
  builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                  random_forest=True)

  # Iterate over individual trees
  for i in range(sklearn_model.n_estimators):
    # Process the i-th tree and add to the builder
    # process_tree() to be defined later
    builder.append( process_tree(sklearn_model.estimators_[i].tree_,
                                 sklearn_model) )

  return builder.commit()

def process_leaf_node(treelite_tree, sklearn_tree, node_id, sklearn_model):
  # The `value` attribute stores the output for every leaf node.
  leaf_value = sklearn_tree.value[node_id].squeeze()
  # Initialize the leaf node with given node ID
  treelite_tree[node_id].set_leaf_node(leaf_value)
