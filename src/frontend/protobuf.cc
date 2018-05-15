/*!
 * Copyright 2017 by Contributors
 * \file protobuf.cc
 * \brief Frontend for protobuf model
 * \author Philip Cho
 */

#include <treelite/tree.h>
#include <queue>
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

}  // namespace anonymous

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(protobuf);

Model LoadProtobufModel(const char* filename) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename, "r"));
  dmlc::istream is(fi.get());
  treelite_protobuf::Model protomodel;
  CHECK(protomodel.ParseFromIstream(&is)) << "Ill-formed Protocol Buffers file";

  Model model;
  CHECK(protomodel.has_num_feature()) << "num_feature must exist";
  const auto num_feature = protomodel.num_feature();
  CHECK_LT(num_feature, std::numeric_limits<int>::max())
    << "num_feature too big";
  CHECK_GT(num_feature, 0) << "num_feature must be positive";
  model.num_feature = static_cast<int>(protomodel.num_feature());

  CHECK(protomodel.has_num_output_group()) << "num_output_group must exist";
  const auto num_output_group = protomodel.num_output_group();
  CHECK_LT(num_output_group, std::numeric_limits<int>::max())
    << "num_output_group too big";
  CHECK_GT(num_output_group, 0) << "num_output_group must be positive";
  model.num_output_group = static_cast<int>(protomodel.num_output_group());

  CHECK(protomodel.has_random_forest_flag())
    << "random_forest_flag must exist";
  model.random_forest_flag = protomodel.random_forest_flag();

  // extra parameters field
  const auto& ep = protomodel.extra_params();
  std::vector<std::pair<std::string, std::string>> cfg;
  std::copy(ep.begin(), ep.end(), std::back_inserter(cfg));
  InitParamAndCheck(&model.param, cfg);

  // flag to check consistent use of leaf vector
  // 0: no leaf should use leaf vector
  // 1: every leaf should use leaf vector
  // -1: indeterminate
  int8_t flag_leaf_vector = -1;

  const int ntree = protomodel.trees_size();
  for (int i = 0; i < ntree; ++i) {
    model.trees.emplace_back();
    Tree& tree = model.trees.back();
    tree.Init();

    CHECK(protomodel.trees(i).has_head());
    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<const treelite_protobuf::Node&, int>> Q;
      // (proto node, ID)
    Q.push({protomodel.trees(i).head(), 0});
    while (!Q.empty()) {
      auto elem = Q.front(); Q.pop();
      const treelite_protobuf::Node& node = elem.first;
      int id = elem.second;
      const NodeType node_type = GetNodeType(node);
      if (node_type == NodeType::kLeaf) {  // leaf node
        CHECK(flag_leaf_vector != 1)
          << "Inconsistent use of leaf vector: if one leaf node does not use"
          << "a leaf vector, *no other* leaf node can use a leaf vector";
        flag_leaf_vector = 0;  // now no leaf can use leaf vector

        tree[id].set_leaf(static_cast<tl_float>(node.leaf_value()));
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
        tree[id].set_leaf_vector(leaf_vector);
      } else if (node_type == NodeType::kNumericalSplit) {  // numerical split
        const auto split_index = node.split_index();
        const std::string opname = node.op();
        CHECK_LT(split_index, model.num_feature)
          << "split_index must be between 0 and [num_feature] - 1.";
        CHECK_GE(split_index, 0) << "split_index must be positive.";
        CHECK_GT(optable.count(opname), 0) << "No operator `"
                                           << opname << "\" exists";
        tree.AddChilds(id);
        tree[id].set_numerical_split(static_cast<unsigned>(split_index),
                             static_cast<tl_float>(node.threshold()),
                             node.default_left(),
                             optable.at(opname.c_str()));
        Q.push({node.left_child(), tree[id].cleft()});
        Q.push({node.right_child(), tree[id].cright()});
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
        tree.AddChilds(id);
        tree[id].set_categorical_split(static_cast<unsigned>(split_index),
                                       node.default_left(),
                                       left_categories);
        Q.push({node.left_child(), tree[id].cleft()});
        Q.push({node.right_child(), tree[id].cright()});
      }
      /* set node statistics */
      if (node.has_data_count()) {
        tree[id].set_data_count(static_cast<size_t>(node.data_count()));
      }
      if (node.has_sum_hess()) {
        tree[id].set_sum_hess(node.sum_hess());
      }
      if (node.has_gain()) {
        tree[id].set_gain(node.gain());
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

}  // namespace frontend
}  // namespace treelite
