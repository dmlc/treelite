# process_tree(), process_node(), process_test_node() omitted to save space
# See the first section for their definitions

def process_model(sklearn_model):
  # Must specify num_output_group and pred_transform
  builder = treelite.ModelBuilder(num_feature=sklearn_model.n_features_,
                                  num_output_group=sklearn_model.n_classes_,
                                  random_forest=True,
                                  pred_transform='identity_multiclass')
  for i in range(sklearn_model.n_estimators):
    # Process i-th tree and add to the builder
    builder.append( process_tree(sklearn_model.estimators_[i].tree_,
                                 sklearn_model) )

  return builder.commit()

def process_leaf_node(treelite_tree, sklearn_tree, node_id, sklearn_model):
  # Get counts for each label class at this leaf node
  leaf_count = sklearn_tree.value[node_id].squeeze()
  # Compute the probability distribution over label classes
  prob_distribution = leaf_count / leaf_count.sum()
  # The leaf output is the probability distribution
  treelite_tree[node_id].set_leaf_node(prob_distribution)
