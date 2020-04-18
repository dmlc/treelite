/*!
 * Copyright 2017 by Contributors
 * \file protobuf.cc
 * \brief Frontend for protobuf model
 * \author Philip Cho
 */

#include <dmlc/logging.h>
#include <treelite/tree.h>
#include <queue>

#ifdef TREELITE_PROTOBUF_SUPPORT

#include "tree.pb.h"

namespace {

enum class NodeType : int8_t {
  kLeaf, kLeafVector, kNumericalSplit, kCategoricalSplit
};

inline NodeType GetNodeType(const treelite_protobuf::Node& node) {
  if (node.has_left_child()) {  // node is non-leaf
    CHECK(node.has_right_child());
    CHECK(node.has_default_left());
    CHECK(node.has_split_index());
    CHECK(node.has_split_type());
    CHECK(!node.has_leaf_value());
    CHECK_EQ(node.leaf_vector_size(), 0);
    const auto split_type = node.split_type();
    if (split_type == treelite_protobuf::Node_SplitFeatureType_NUMERICAL) {
      // numerical split
      CHECK(node.has_op());
      CHECK(node.has_threshold());
      CHECK_EQ(node.left_categories_size(), 0);
      return NodeType::kNumericalSplit;
    } else {  // categorical split
      CHECK(!node.has_op());
      CHECK(!node.has_threshold());
      CHECK(node.has_missing_category_to_zero());
      return NodeType::kCategoricalSplit;
    }
  } else {  // node is leaf
    CHECK(!node.has_right_child());
    CHECK(!node.has_default_left());
    CHECK(!node.has_split_index());
    CHECK(!node.has_split_type());
    CHECK(!node.has_op());
    CHECK(!node.has_threshold());
    CHECK(!node.has_gain());
    CHECK(!node.has_missing_category_to_zero());
    CHECK_EQ(node.left_categories_size(), 0);
    if (node.has_leaf_value()) {
      CHECK_EQ(node.leaf_vector_size(), 0);
      return NodeType::kLeaf;
    } else {
      CHECK_GT(node.leaf_vector_size(), 0);
      return NodeType::kLeafVector;
    }
  }
}

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(protobuf);

Model LoadProtobufModel(const char* filename) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename, "r"));
  dmlc::istream is(fi.get());
  treelite_protobuf::Model pb_model;
  CHECK(pb_model.ParseFromIstream(&is)) << "Ill-formed Protocol Buffers file";

  Model model;
  CHECK(pb_model.has_num_feature()) << "num_feature must exist";
  const auto num_feature = pb_model.num_feature();
  CHECK_LT(num_feature, std::numeric_limits<int>::max())
    << "num_feature too big";
  CHECK_GT(num_feature, 0) << "num_feature must be positive";
  model.num_feature = static_cast<int>(pb_model.num_feature());

  CHECK(pb_model.has_num_output_group()) << "num_output_group must exist";
  const auto num_output_group = pb_model.num_output_group();
  CHECK_LT(num_output_group, std::numeric_limits<int>::max())
    << "num_output_group too big";
  CHECK_GT(num_output_group, 0) << "num_output_group must be positive";
  model.num_output_group = static_cast<int>(pb_model.num_output_group());

  CHECK(pb_model.has_random_forest_flag())
    << "random_forest_flag must exist";
  model.random_forest_flag = pb_model.random_forest_flag();

  // extra parameters field
  const auto& ep = pb_model.extra_params();
  std::vector<std::pair<std::string, std::string>> cfg;
  std::copy(ep.begin(), ep.end(), std::back_inserter(cfg));
  InitParamAndCheck(&model.param, cfg);

  // flag to check consistent use of leaf vector
  // 0: no leaf should use leaf vector
  // 1: every leaf should use leaf vector
  // -1: indeterminate
  int8_t flag_leaf_vector = -1;

  const int ntree = pb_model.trees_size();
  for (int i = 0; i < ntree; ++i) {
    model.trees.emplace_back();
    Tree& tree = model.trees.back();
    tree.Init();

    const treelite_protobuf::Tree& pb_tree = pb_model.trees(i);
    const int num_node = pb_tree.nodes_size();
    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
    Q.push({0, 0});
    while (!Q.empty()) {
      int old_id, new_id;
      std::tie(old_id, new_id) = Q.front(); Q.pop();
      const treelite_protobuf::Node& node = pb_tree.nodes(old_id);
      const NodeType node_type = GetNodeType(node);
      if (node_type == NodeType::kLeaf) {  // leaf node with a scalar output
        CHECK(flag_leaf_vector != 1)
          << "Inconsistent use of leaf vector: if one leaf node does not use"
          << "a leaf vector, *no other* leaf node can use a leaf vector";
        flag_leaf_vector = 0;  // now no leaf can use leaf vector
        tree[new_id].set_leaf(static_cast<tl_float>(node.leaf_value()));
      } else if (node_type == NodeType::kLeafVector) {
        // leaf node with vector output
        CHECK(flag_leaf_vector != 0)
          << "Inconsistent use of leaf vector: if one leaf node uses "
          << "a leaf vector, *every* leaf node must use a leaf vector as well";
        flag_leaf_vector = 1;  // now every leaf must use leaf vector

        const int len = node.leaf_vector_size();
        CHECK_EQ(len, model.num_output_group)
          << "The length of leaf vector must be identical to the "
          << "number of output groups";
        std::vector<tl_float> leaf_vector(len);
        for (int i = 0; i < len; ++i) {
          leaf_vector[i] = static_cast<tl_float>(node.leaf_vector(i));
        }
        tree[new_id].set_leaf_vector(leaf_vector);
      } else if (node_type == NodeType::kNumericalSplit) {  // numerical split
        const auto split_index = node.split_index();
        const std::string opname = node.op();
        CHECK_LT(split_index, model.num_feature)
          << "split_index must be between 0 and [num_feature] - 1.";
        CHECK_GE(split_index, 0) << "split_index must be positive.";
        CHECK_GT(optable.count(opname), 0) << "No operator `"
                                           << opname << "\" exists";
        tree.AddChilds(new_id);
        tree[new_id].set_numerical_split(static_cast<unsigned>(split_index),
                                         static_cast<tl_float>(node.threshold()),
                                         node.default_left(),
                                         optable.at(opname.c_str()));
        Q.push({node.left_child(), tree[new_id].cleft()});
        Q.push({node.right_child(), tree[new_id].cright()});
      } else {  // categorical split
        const auto split_index = node.split_index();
        CHECK_LT(split_index, model.num_feature)
          << "split_index must be between 0 and [num_feature] - 1.";
        CHECK_GE(split_index, 0) << "split_index must be positive.";
        const int left_categories_size = node.left_categories_size();
        std::vector<uint32_t> left_categories;
        for (int i = 0; i < left_categories_size; ++i) {
          const auto cat = node.left_categories(i);
          CHECK(cat <= std::numeric_limits<uint32_t>::max());
          left_categories.push_back(static_cast<uint32_t>(cat));
        }
        tree.AddChilds(new_id);
        tree[new_id].set_categorical_split(static_cast<unsigned>(split_index),
                                           node.default_left(),
                                           node.missing_category_to_zero(),
                                           left_categories);
        Q.push({node.left_child(), tree[new_id].cleft()});
        Q.push({node.right_child(), tree[new_id].cright()});
      }
      /* set node statistics */
      if (node.has_data_count()) {
        tree[new_id].set_data_count(static_cast<size_t>(node.data_count()));
      }
      if (node.has_sum_hess()) {
        tree[new_id].set_sum_hess(node.sum_hess());
      }
      if (node.has_gain()) {
        tree[new_id].set_gain(node.gain());
      }
    }
  }
  if (flag_leaf_vector == 0) {
    if (model.num_output_group > 1) {
      // multiclass classification with gradient boosted trees
      CHECK(!model.random_forest_flag)
        << "To use a random forest for multi-class classification, each leaf "
        << "node must output a leaf vector specifying a probability "
        << "distribution";
      CHECK_EQ(ntree % model.num_output_group, 0)
        << "For multi-class classifiers with gradient boosted trees, the number "
        << "of trees must be evenly divisible by the number of output groups";
    }
  } else if (flag_leaf_vector == 1) {
    // multiclass classification with a random forest
    CHECK(model.random_forest_flag)
      << "In multi-class classifiers with gradient boosted trees, each leaf "
      << "node must output a single floating-point value.";
  } else {
    LOG(FATAL) << "Impossible thing happened: model has no leaf node!";
  }
  return model;
}

void ExportProtobufModel(const char* filename, const Model& model) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename, "w"));
  dmlc::ostream os(fi.get());
  treelite_protobuf::Model pb_model;

  pb_model.set_num_feature(
    static_cast<google::protobuf::int32>(model.num_feature));

  pb_model.set_num_output_group(
    static_cast<google::protobuf::int32>(model.num_output_group));

  pb_model.set_random_forest_flag(model.random_forest_flag);

  // extra parameters field
  for (const auto& kv : model.param.__DICT__()) {
    (*pb_model.mutable_extra_params())[kv.first] = kv.second;
  }

  // flag to check consistent use of leaf vector
  // 0: no leaf should use leaf vector
  // 1: every leaf should use leaf vector
  // -1: indeterminate
  int8_t flag_leaf_vector = -1;

  const int ntree = model.trees.size();
  for (int i = 0; i < ntree; ++i) {
    const Tree& tree = model.trees[i];
    treelite_protobuf::Tree* pb_tree = pb_model.add_trees();

    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      treelite_protobuf::Node* pb_node = pb_tree->add_nodes();
      if (tree[nid].is_leaf()) {  // leaf node
        if (tree[nid].has_leaf_vector()) {  // leaf node with vector output
          CHECK(flag_leaf_vector != 0)
            << "Inconsistent use of leaf vector: if one leaf node uses "
            << "a leaf vector, *every* leaf node must use a leaf vector as well";
          flag_leaf_vector = 1;  // now every leaf must use leaf vector

          const auto& leaf_vector = tree[nid].leaf_vector();
          CHECK_EQ(leaf_vector.size(), model.num_output_group)
            << "The length of leaf vector must be identical to the "
            << "number of output groups";
          for (tl_float e : leaf_vector) {
            pb_node->add_leaf_vector(static_cast<double>(e));
          }
          CHECK_EQ(pb_node->leaf_vector_size(), leaf_vector.size());
        } else {  // leaf node with scalar output
          CHECK(flag_leaf_vector != 1)
            << "Inconsistent use of leaf vector: if one leaf node does not use"
            << "a leaf vector, *no other* leaf node can use a leaf vector";
          flag_leaf_vector = 0;  // now no leaf can use leaf vector
          pb_node->set_leaf_value(static_cast<double>(tree[nid].leaf_value()));
        }
      } else if (tree[nid].split_type() == SplitFeatureType::kNumerical) {
        // numerical split
        const unsigned split_index = tree[nid].split_index();
        const tl_float threshold = tree[nid].threshold();
        const bool default_left = tree[nid].default_left();
        const Operator op = tree[nid].comparison_op();

        pb_node->set_default_left(default_left);
        pb_node->set_split_index(static_cast<google::protobuf::int32>(split_index));
        pb_node->set_split_type(treelite_protobuf::Node_SplitFeatureType_NUMERICAL);
        pb_node->set_op(OpName(op));
        pb_node->set_threshold(static_cast<double>(threshold));
        pb_node->set_left_child(tree[nid].cleft());
        pb_node->set_right_child(tree[nid].cright());
      } else {  // categorical split
        const unsigned split_index = tree[nid].split_index();
        const auto& left_categories = tree[nid].left_categories();
        const bool default_left = tree[nid].default_left();
        const bool missing_category_to_zero = tree[nid].missing_category_to_zero();

        pb_node->set_default_left(default_left);
        pb_node->set_split_index(static_cast<google::protobuf::int32>(split_index));
        pb_node->set_split_type(treelite_protobuf::Node_SplitFeatureType_CATEGORICAL);
        pb_node->set_missing_category_to_zero(missing_category_to_zero);
        for (auto e : left_categories) {
          pb_node->add_left_categories(static_cast<google::protobuf::uint32>(e));
        }
        pb_node->set_left_child(tree[nid].cleft());
        pb_node->set_right_child(tree[nid].cright());
      }
      /* set node statistics */
      if (tree[nid].has_data_count()) {
        pb_node->set_data_count(
          static_cast<google::protobuf::uint64>(tree[nid].data_count()));
      }
      if (tree[nid].has_sum_hess()) {
        pb_node->set_sum_hess(tree[nid].sum_hess());
      }
      if (tree[nid].has_gain()) {
        pb_node->set_gain(tree[nid].gain());
      }
    }
  }
  CHECK(pb_model.SerializeToOstream(&os))
    << "Failed to write Protocol Buffers file";
  os.set_stream(nullptr);
}

}  // namespace frontend
}  // namespace treelite

#else   // TREELITE_PROTOBUF_SUPPORT

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(protobuf);

Model LoadProtobufModel(const char* filename) {
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
  return Model();  // should not reach here
}

void ExportProtobufModel(const char* filename, const Model& model) {
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
}

}  // namespace frontend
}  // namespace treelite

#endif  // TREELITE_PROTOBUF_SUPPORT
