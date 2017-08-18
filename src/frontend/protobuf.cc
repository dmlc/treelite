/*!
 * Copyright 2017 by Contributors
 * \file protobuf.cc
 * \brief Frontend for protobuf model
 * \author Philip Cho
 */

#include <treelite/tree.h>
#include <queue>

#ifdef PROTOBUF_SUPPORT
#include "tree.pb.h"
namespace {

enum class NodeType : int8_t {
  kLeaf, kNumericalSplit, kCategoricalSplit
};

inline NodeType GetNodeType(const treelite_protobuf::Node& node) {
  if (node.has_left_child()) {  // node is non-leaf
    CHECK(node.has_right_child());
    CHECK(node.has_default_left());
    CHECK(node.has_split_index());
    CHECK(node.has_split_type());
    CHECK(!node.has_leaf_value());
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
    CHECK_EQ(node.left_categories_size(), 0);
    CHECK(node.has_leaf_value());
    return NodeType::kLeaf;
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
  CHECK(protomodel.has_num_features());
  const auto num_features = protomodel.num_features();
  CHECK_LT(num_features, std::numeric_limits<int>::max())
    << "num_features too big";
  CHECK_GT(num_features, 0) << "num_features must be positive";
  model.num_features = static_cast<int>(protomodel.num_features());
  // extra parameters field
  const auto& ep = protomodel.extra_params();
  std::vector<std::pair<std::string, std::string>> cfg;
  std::copy(ep.begin(), ep.end(), std::back_inserter(cfg));
  InitParamAndCheck(&model.param, cfg);

  const int ntree = protomodel.trees_size();
  for (int i = 0; i < ntree; ++i) {
    model.trees.emplace_back();
    treelite::Tree& tree = model.trees.back();
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
      if (node_type == NodeType::kLeaf) {
        tree[id].set_leaf(static_cast<treelite::tl_float>(node.leaf_value()));
      } else if (node_type == NodeType::kNumericalSplit) {
        const auto split_index = node.split_index();
        const std::string opname = node.op();
        CHECK_LT(split_index, model.num_features)
          << "split_index must be between 0 and [num_featues]-1.";
        CHECK_GE(split_index, 0) << "split_index must be positive.";
        CHECK_GT(optable.count(opname), 0) << "No operator `"
                                           << opname << "\" exists";
        tree.AddChilds(id);
        tree[id].set_numerical_split(static_cast<unsigned>(split_index),
                             static_cast<treelite::tl_float>(node.threshold()),
                             node.default_left(),
                             optable.at(opname.c_str()));
        Q.push({node.left_child(), tree[id].cleft()});
        Q.push({node.right_child(), tree[id].cright()});
      } else {  // categorical split
        const auto split_index = node.split_index();
        CHECK_LT(split_index, model.num_features)
          << "split_index must be between 0 and [num_featues]-1.";
        CHECK_GE(split_index, 0) << "split_index must be positive.";
        const int left_categories_size = node.left_categories_size();
        std::vector<uint8_t> left_categories;
        for (int i = 0; i < left_categories_size; ++i) {
          const auto cat = node.left_categories(i);
          CHECK(cat <= std::numeric_limits<uint8_t>::max());
          left_categories.push_back(static_cast<uint8_t>(cat));
        }
        tree.AddChilds(id);
        tree[id].set_categorical_split(static_cast<unsigned>(split_index),
                                       node.default_left(),
                                       left_categories);
        Q.push({node.left_child(), tree[id].cleft()});
        Q.push({node.right_child(), tree[id].cright()});
      }
    }
  }
  return model;
}

}  // namespace frontend
}  // namespace treelite

#else

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(protobuf);

Model LoadProtobufModel(const char* filename) {
  LOG(FATAL) << "Protobuf library not linked";
  return Model();
}

}  // namespace frontend
}  // napespace treelite

#endif  // PROTOBUF_SUPPORT
