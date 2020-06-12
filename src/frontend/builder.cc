/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file builder.cc
 * \brief model builder frontend
 * \author Hyunsu Cho
 */

#include <dmlc/registry.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <memory>
#include <queue>
#include "../c_api/c_api_error.h"

#define CHECK_EARLY_RETURN(x, msg)                           \
  if (!(x)) {                                                \
    TreeliteAPISetLastError(msg); \
    dmlc::LogMessage(__FILE__, __LINE__).stream() << msg;    \
    return false;                                            \
  }

/* data structures with underscore prefixes are internal use only and
   do not have external linkage */
namespace {

struct _Node {
  enum class _Status : int8_t {
    kEmpty, kNumericalTest, kCategoricalTest, kLeaf
  };
  union _Info {
    treelite::tl_float leaf_value;  // for leaf nodes
    treelite::tl_float threshold;   // for non-leaf nodes
  };
  /*
   * leaf vector: only used for random forests with multi-class classification
   */
  std::vector<treelite::tl_float> leaf_vector;
  _Status status;
  /* pointers to parent, left and right children */
  _Node* parent;
  _Node* left_child;
  _Node* right_child;
  // split feature index
  unsigned feature_id;
  // default direction for missing values
  bool default_left;
  // extra info: leaf value or threshold
  _Info info;
  // (for numerical split)
  // operator to use for expression of form [fval] OP [threshold]
  // If the expression evaluates to true, take the left child;
  // otherwise, take the right child.
  treelite::Operator op;
  // (for categorical split)
  // list of all categories that belong to the left child node.
  // All others not in the list belong to the right child node.
  // Categories are integers ranging from 0 to (n-1), where n is the number of
  // categories in that particular feature. Let's assume n <= 64.
  std::vector<uint32_t> left_categories;

  inline _Node()
    : status(_Status::kEmpty),
      parent(nullptr), left_child(nullptr), right_child(nullptr) {}
};

struct _Tree {
  _Node* root;
  std::unordered_map<int, std::shared_ptr<_Node>> nodes;
  inline _Tree() : root(nullptr), nodes() {}
};

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(builder);

struct TreeBuilderImpl {
  _Tree tree;
  inline TreeBuilderImpl() : tree() {}
};

struct ModelBuilderImpl {
  std::vector<TreeBuilder> trees;
  int num_feature;
  int num_output_group;
  bool random_forest_flag;
  std::vector<std::pair<std::string, std::string>> cfg;
  inline ModelBuilderImpl(int num_feature, int num_output_group,
                          bool random_forest_flag)
    : trees(), num_feature(num_feature),
      num_output_group(num_output_group),
      random_forest_flag(random_forest_flag), cfg() {
    CHECK_GT(num_feature, 0) << "ModelBuilder: num_feature must be positive";
    CHECK_GT(num_output_group, 0)
      << "ModelBuilder: num_output_group must be positive";
  }
};

TreeBuilder::TreeBuilder()
  : pimpl(new TreeBuilderImpl()), ensemble_id(nullptr) {}
TreeBuilder::~TreeBuilder() {}

bool
TreeBuilder::CreateNode(int node_key) {
  auto& nodes = pimpl->tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) == 0,
                     "CreateNode: nodes with duplicate keys are not allowed");
  nodes[node_key].reset(new _Node());
  return true;
}

bool
TreeBuilder::DeleteNode(int node_key) {
  auto& tree = pimpl->tree;
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "DeleteNode: no node found with node_key");
  _Node* node = nodes[node_key].get();
  if (tree.root == node) {  // deleting root
    tree.root = nullptr;
  }
  if (node->left_child != nullptr) {  // deleting a parent
    node->left_child->parent = nullptr;
  }
  if (node->right_child != nullptr) {  // deleting a parent
    node->right_child->parent = nullptr;
  }
  nodes.erase(node_key);
  return true;
}

bool
TreeBuilder::SetRootNode(int node_key) {
  auto& tree = pimpl->tree;
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "SetRootNode: no node found with node_key");
  _Node* node = nodes[node_key].get();
  CHECK_EARLY_RETURN(node->parent == nullptr,
                     "SetRootNode: a root node cannot have a parent");
  tree.root = node;
  return true;
}

bool
TreeBuilder::SetNumericalTestNode(int node_key,
                                  unsigned feature_id,
                                  Operator op, tl_float threshold,
                                  bool default_left, int left_child_key,
                                  int right_child_key) {
  auto& tree = pimpl->tree;
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "SetNumericalTestNode: no node found with node_key");
  CHECK_EARLY_RETURN(nodes.count(left_child_key) > 0,
                    "SetNumericalTestNode: no node found with left_child_key");
  CHECK_EARLY_RETURN(nodes.count(right_child_key) > 0,
                   "SetNumericalTestNode: no node found with right_child_key");
  _Node* node = nodes[node_key].get();
  _Node* left_child = nodes[left_child_key].get();
  _Node* right_child = nodes[right_child_key].get();
  CHECK_EARLY_RETURN(node->status == _Node::_Status::kEmpty,
                     "SetNumericalTestNode: cannot modify a non-empty node");
  CHECK_EARLY_RETURN(left_child->parent == nullptr,
             "SetNumericalTestNode: node designated as left child already has "
             "a parent");
  CHECK_EARLY_RETURN(right_child->parent == nullptr,
            "SetNumericalTestNode: node designated as right child already has "
            "a parent");
  CHECK_EARLY_RETURN(left_child != tree.root && right_child != tree.root,
                     "SetNumericalTestNode: the root node cannot be a child");
  node->status = _Node::_Status::kNumericalTest;
  node->left_child = nodes[left_child_key].get();
  node->left_child->parent = node;
  node->right_child = nodes[right_child_key].get();
  node->right_child->parent = node;
  node->feature_id = feature_id;
  node->default_left = default_left;
  node->info.threshold = threshold;
  node->op = op;
  return true;
}

bool
TreeBuilder::SetCategoricalTestNode(int node_key,
                                    unsigned feature_id,
                                    const std::vector<uint32_t>& left_categories,
                                    bool default_left, int left_child_key,
                                    int right_child_key) {
  auto& tree = pimpl->tree;
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "SetCategoricalTestNode: no node found with node_key");
  CHECK_EARLY_RETURN(nodes.count(left_child_key) > 0,
                  "SetCategoricalTestNode: no node found with left_child_key");
  CHECK_EARLY_RETURN(nodes.count(right_child_key) > 0,
                 "SetCategoricalTestNode: no node found with right_child_key");
  _Node* node = nodes[node_key].get();
  _Node* left_child = nodes[left_child_key].get();
  _Node* right_child = nodes[right_child_key].get();
  CHECK_EARLY_RETURN(node->status == _Node::_Status::kEmpty,
                     "SetCategoricalTestNode: cannot modify a non-empty node");
  CHECK_EARLY_RETURN(left_child->parent == nullptr,
               "SetCategoricalTestNode: node designated as left child already "
               "has a parent");
  CHECK_EARLY_RETURN(right_child->parent == nullptr,
              "SetCategoricalTestNode: node designated as right child already "
              "has a parent");
  CHECK_EARLY_RETURN(left_child != tree.root && right_child != tree.root,
                    "SetCategoricalTestNode: the root node cannot be a child");
  node->status = _Node::_Status::kCategoricalTest;
  node->left_child = nodes[left_child_key].get();
  node->left_child->parent = node;
  node->right_child = nodes[right_child_key].get();
  node->right_child->parent = node;
  node->feature_id = feature_id;
  node->default_left = default_left;
  node->left_categories = left_categories;
  return true;
}

bool
TreeBuilder::SetLeafNode(int node_key, tl_float leaf_value) {
  auto& tree = pimpl->tree;
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "SetLeafNode: no node found with node_key");
  _Node* node = nodes[node_key].get();
  CHECK_EARLY_RETURN(node->status == _Node::_Status::kEmpty,
                     "SetLeafNode: cannot modify a non-empty node");
  node->status = _Node::_Status::kLeaf;
  node->info.leaf_value = leaf_value;
  return true;
}

bool
TreeBuilder::SetLeafVectorNode(int node_key,
                               const std::vector<tl_float>& leaf_vector) {
  auto& tree = pimpl->tree;
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
    "SetLeafVectorNode: no node found with node_key");
  _Node* node = nodes[node_key].get();
  CHECK_EARLY_RETURN(node->status == _Node::_Status::kEmpty,
    "SetLeafVectorNode: cannot modify a non-empty node");
  node->status = _Node::_Status::kLeaf;
  node->leaf_vector = leaf_vector;
  return true;
}

ModelBuilder::ModelBuilder(int num_feature, int num_output_group,
                           bool random_forest_flag)
  : pimpl(new ModelBuilderImpl(num_feature,
                               num_output_group,
                               random_forest_flag)) {}
ModelBuilder::~ModelBuilder() {}

void
ModelBuilder::SetModelParam(const char* name, const char* value) {
  pimpl->cfg.emplace_back(name, value);
}

int
ModelBuilder::InsertTree(TreeBuilder* tree_builder, int index) {
  if (tree_builder == nullptr) {
    const char* msg = "InsertTree: not a valid tree builder";
    LOG(INFO) << msg;
    TreeliteAPISetLastError(msg);
    return -1;  // fail
  }
  if (tree_builder->ensemble_id != nullptr) {
    const char* msg = "InsertTree: tree is already part of another ensemble";
    LOG(INFO) << msg;
    TreeliteAPISetLastError(msg);
    return -1;  // fail
  }

  // check bounds for feature indices
  for (const auto& kv : tree_builder->pimpl->tree.nodes) {
    const _Node::_Status status = kv.second->status;
    if (status == _Node::_Status::kNumericalTest ||
      status == _Node::_Status::kCategoricalTest) {
      const int fid = static_cast<int>(kv.second->feature_id);
      if (fid < 0 || fid >= pimpl->num_feature) {
        std::ostringstream oss;
        oss << "InsertTree: tree has an invalid split at node "
          << kv.first << ": feature id " << kv.second->feature_id
          << " is out of bound";
        const std::string str = oss.str();
        const char* msg = str.c_str();
        LOG(INFO) << msg;
        TreeliteAPISetLastError(msg);
        return -1;  // fail
      }
    }
  }

  // perform insertion
  auto& trees = pimpl->trees;
  if (index == -1) {
    trees.push_back(std::move(*tree_builder));
    tree_builder->ensemble_id = static_cast<void*>(this);
    return static_cast<int>(trees.size());
  } else {
    if (static_cast<size_t>(index) <= trees.size()) {
      trees.insert(trees.begin() + index, std::move(*tree_builder));
      tree_builder->ensemble_id = static_cast<void*>(this);
      return index;
    } else {
      LOG(INFO) << "CreateTree: index out of bound";
      return -1;  // fail
    }
  }
}

TreeBuilder&
ModelBuilder::GetTree(int index) {
  return pimpl->trees[index];
}

const TreeBuilder&
ModelBuilder::GetTree(int index) const {
  return pimpl->trees[index];
}

bool
ModelBuilder::DeleteTree(int index) {
  auto& trees = pimpl->trees;
  CHECK_EARLY_RETURN(static_cast<size_t>(index) < trees.size(),
                     "DeleteTree: index out of bound");
  trees.erase(trees.begin() + index);
  return true;
}

bool
ModelBuilder::CommitModel(Model* out_model) {
  Model model;
  model.num_feature = pimpl->num_feature;
  model.num_output_group = pimpl->num_output_group;
  model.random_forest_flag = pimpl->random_forest_flag;
  // extra parameters
  InitParamAndCheck(&model.param, pimpl->cfg);

  // flag to check consistent use of leaf vector
  // 0: no leaf should use leaf vector
  // 1: every leaf should use leaf vector
  // -1: indeterminate
  int8_t flag_leaf_vector = -1;

  for (const auto& _tree_builder : pimpl->trees) {
    const auto& _tree = _tree_builder.pimpl->tree;
    CHECK_EARLY_RETURN(_tree.root != nullptr,
                       "CommitModel: a tree has no root node");
    model.trees.emplace_back();
    Tree& tree = model.trees.back();
    tree.Init();

    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<const _Node*, int>> Q;  // (internal pointer, ID)
    Q.push({_tree.root, 0});  // assign 0 to root
    while (!Q.empty()) {
      const _Node* node;
      int nid;
      std::tie(node, nid) = Q.front(); Q.pop();
      CHECK_EARLY_RETURN(node->status != _Node::_Status::kEmpty,
             "CommitModel: encountered an empty node in the middle of a tree");
      if (node->status == _Node::_Status::kNumericalTest) {
        CHECK_EARLY_RETURN(node->left_child != nullptr,
                           "CommitModel: a test node lacks a left child");
        CHECK_EARLY_RETURN(node->right_child != nullptr,
                           "CommitModel: a test node lacks a right child");
        CHECK_EARLY_RETURN(node->left_child->parent == node,
                           "CommitModel: left child has wrong parent");
        CHECK_EARLY_RETURN(node->right_child->parent == node,
                           "CommitModel: right child has wrong parent");
        tree.AddChilds(nid);
        tree.SetNumericalSplit(nid, node->feature_id, node->info.threshold,
                               node->default_left, node->op);
        Q.push({node->left_child, tree.LeftChild(nid)});
        Q.push({node->right_child, tree.RightChild(nid)});
      } else if (node->status == _Node::_Status::kCategoricalTest) {
        CHECK_EARLY_RETURN(node->left_child != nullptr,
                           "CommitModel: a test node lacks a left child");
        CHECK_EARLY_RETURN(node->right_child != nullptr,
                           "CommitModel: a test node lacks a right child");
        CHECK_EARLY_RETURN(node->left_child->parent == node,
                           "CommitModel: left child has wrong parent");
        CHECK_EARLY_RETURN(node->right_child->parent == node,
                           "CommitModel: right child has wrong parent");
        tree.AddChilds(nid);
        tree.SetCategoricalSplit(nid, node->feature_id, node->default_left,
                                 false, node->left_categories);
        Q.push({node->left_child, tree.LeftChild(nid)});
        Q.push({node->right_child, tree.RightChild(nid)});
      } else {  // leaf node
        CHECK_EARLY_RETURN(node->left_child == nullptr
                           && node->right_child == nullptr,
                           "CommitModel: a leaf node cannot have children");
        if (!node->leaf_vector.empty()) {  // leaf vector exists
          CHECK_EARLY_RETURN(flag_leaf_vector != 0,
                             "CommitModel: Inconsistent use of leaf vector: "
                             "if one leaf node uses a leaf vector, "
                             "*every* leaf node must use a leaf vector");
          flag_leaf_vector = 1;  // now every leaf must use leaf vector
          CHECK_EARLY_RETURN(node->leaf_vector.size() == model.num_output_group,
                              "CommitModel: The length of leaf vector must be "
                              "identical to the number of output groups");
          tree.SetLeafVector(nid, node->leaf_vector);
        } else {  // ordinary leaf
          CHECK_EARLY_RETURN(flag_leaf_vector != 1,
                             "CommitModel: Inconsistent use of leaf vector: "
                             "if one leaf node does not use a leaf vector, "
                             "*no other* leaf node can use a leaf vector");
          flag_leaf_vector = 0;  // now no leaf can use leaf vector
          tree.SetLeaf(nid, node->info.leaf_value);
        }
      }
    }
  }
  if (flag_leaf_vector == 0) {
    if (model.num_output_group > 1) {
      // multiclass classification with gradient boosted trees
      CHECK_EARLY_RETURN(!model.random_forest_flag,
        "To use a random forest for multi-class classification, each leaf "
        "node must output a leaf vector specifying a probability "
        "distribution");
      CHECK_EARLY_RETURN(pimpl->trees.size() % model.num_output_group == 0,
        "For multi-class classifiers with gradient boosted trees, the number "
        "of trees must be evenly divisible by the number of output groups");
    }
  } else if (flag_leaf_vector == 1) {
    // multiclass classification with a random forest
    CHECK_EARLY_RETURN(model.random_forest_flag,
      "In multi-class classifiers with gradient boosted trees, each leaf "
      "node must output a single floating-point value.");
  } else {
    LOG(FATAL) << "Impossible thing happened: model has no leaf node!";
  }
  *out_model = std::move(model);
  return true;
}

}  // namespace frontend
}  // namespace treelite
