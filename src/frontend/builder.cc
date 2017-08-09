/*!
 * Copyright 2017 by Contributors
 * \file builder.cc
 * \brief model builder frontend
 * \author Philip Cho
 */

#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <dmlc/registry.h>
#include <memory>
#include <queue>

#define CHECK_EARLY_RETURN(x, msg)                           \
  if (!(x)) {                                                \
    dmlc::LogMessage(__FILE__, __LINE__).stream() << msg;    \
    return false;                                            \
  }

/* data structures with underscore prefixes are internal use only and
   do not have external linkage */
namespace {

struct _Node {
  enum class _Status : int8_t {
    kEmpty, kTest, kLeaf
  };
  union _Info {
    treelite::tl_float leaf_value;  // for leaf nodes
    treelite::tl_float threshold;   // for non-leaf nodes
  };
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
  // operator to use for expression of form [fval] OP [threshold]
  // If the expression evaluates to true, take the left child;
  // otherwise, take the right child.
  treelite::Operator op;

  inline _Node()
    : status(_Status::kEmpty),
      parent(nullptr), left_child(nullptr), right_child(nullptr) {}
};

struct _Tree {
  _Node* root;
  std::unordered_map<int, std::unique_ptr<_Node>> nodes;
  inline _Tree() : root(nullptr), nodes() {}
};

}  // namespace anonymous

namespace treelite {
namespace frontend {
namespace builder {

DMLC_REGISTRY_FILE_TAG(builder);

struct ModelBuilderImpl {
  std::vector<_Tree> trees;
  int num_features;
  inline ModelBuilderImpl(int num_features)
    : trees(), num_features(num_features) {
    CHECK_GT(num_features, 0) << "ModelBuilder: num_features must be positive";
  }
};

ModelBuilder::ModelBuilder(int num_features)
  : pimpl(common::make_unique<ModelBuilderImpl>(num_features)) {}
ModelBuilder::~ModelBuilder() {}

int
ModelBuilder::CreateTree(int index) {
  auto& trees = pimpl->trees;
  if (index == -1) {
    trees.push_back(_Tree());
    return static_cast<int>(trees.size());
  } else {
    if (static_cast<size_t>(index) <= trees.size()) {
      trees.insert(trees.begin() + index, _Tree());
      return index;
    } else {
      LOG(INFO) << "CreateTree: index out of bound";
      return -1;  // fail
    }
  }
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
ModelBuilder::CreateNode(int tree_index, int node_key) {
  auto& trees = pimpl->trees;
  CHECK_EARLY_RETURN(static_cast<size_t>(tree_index) < trees.size(),
                     "CreateNode: tree_index out of bound");
  auto& nodes = trees[tree_index].nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) == 0,
                     "CreateNode: nodes with duplicate keys are not allowed");
  nodes[node_key] = common::make_unique<_Node>();
  return true;
}

bool
ModelBuilder::DeleteNode(int tree_index, int node_key) {
  auto& trees = pimpl->trees;
  CHECK_EARLY_RETURN(static_cast<size_t>(tree_index) < trees.size(),
                     "DeleteNode: tree_index out of bound");
  auto& tree = trees[tree_index];
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
ModelBuilder::SetRootNode(int tree_index, int node_key) {
  auto& trees = pimpl->trees;
  CHECK_EARLY_RETURN(static_cast<size_t>(tree_index) < trees.size(),
                     "SetRootNode: tree_index out of bound");
  auto& tree = trees[tree_index];
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "SetRootNode: no node found with node_key");
  _Node* node = nodes[node_key].get();
  CHECK_EARLY_RETURN(node->status != _Node::_Status::kLeaf,
                     "SetRootNode: cannot set a leaf node as root");
  CHECK_EARLY_RETURN(node->parent == nullptr,
                     "SetRootNode: a root node cannot have a parent");
  tree.root = node;
  return true;
}

bool
ModelBuilder::SetTestNode(int tree_index, int node_key, unsigned feature_id,
                          Operator op, tl_float threshold, bool default_left,
                          int left_child_key, int right_child_key) {
  auto& trees = pimpl->trees;
  CHECK_EARLY_RETURN(static_cast<size_t>(tree_index) < trees.size(),
                     "SetTestNode: tree_index out of bound");
  CHECK_EARLY_RETURN(static_cast<int>(feature_id) >= 0 &&
                     static_cast<int>(feature_id) < pimpl->num_features,
                     "SetTestNode: feature id out of bound");
  auto& tree = trees[tree_index];
  auto& nodes = tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) > 0,
                     "SetTestNode: no node found with node_key");
  CHECK_EARLY_RETURN(nodes.count(left_child_key) > 0,
                     "SetTestNode: no node found with left_child_key");
  CHECK_EARLY_RETURN(nodes.count(right_child_key) > 0,
                     "SetTestNode: no node found with right_child_key");
  _Node* node = nodes[node_key].get();
  _Node* left_child = nodes[left_child_key].get();
  _Node* right_child = nodes[right_child_key].get();
  CHECK_EARLY_RETURN(node->status == _Node::_Status::kEmpty,
                     "SetTestNode: cannot modify a non-empty node");
  CHECK_EARLY_RETURN(left_child->parent == nullptr,
            "SetTestNode: node designated as left child already has a parent");
  CHECK_EARLY_RETURN(right_child->parent == nullptr,
            "SetTestNode: node designated as right child already has a parent");
  CHECK_EARLY_RETURN(left_child != tree.root && right_child != tree.root,
                     "SetTestNode: the root node cannot be a child");
  node->status = _Node::_Status::kTest;
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
ModelBuilder::SetLeafNode(int tree_index, int node_key, tl_float leaf_value) {
  auto& trees = pimpl->trees;
  CHECK_EARLY_RETURN(static_cast<size_t>(tree_index) < trees.size(),
                     "SetLeafNode: tree_index out of bound");
  auto& tree = trees[tree_index];
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
ModelBuilder::CommitModel(Model* out_model) {
  Model model;
  model.num_features = pimpl->num_features;
  for (const auto& _tree : pimpl->trees) {
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
      if (node->status == _Node::_Status::kTest) {  // non-leaf node
        CHECK_EARLY_RETURN(node->left_child != nullptr,
                           "CommitModel: a test node lacks a left child");
        CHECK_EARLY_RETURN(node->right_child != nullptr,
                           "CommitModel: a test node lacks a right child");
        CHECK_EARLY_RETURN(node->left_child->parent == node,
                           "CommitModel: left child has wrong parent");
        CHECK_EARLY_RETURN(node->right_child->parent == node,
                           "CommitModel: right child has wrong parent");
        tree.AddChilds(nid);
        tree[nid].set_split(node->feature_id, node->info.threshold,
                            node->default_left, node->op);
        Q.push({node->left_child, tree[nid].cleft()});
        Q.push({node->right_child, tree[nid].cright()});
      } else {  // leaf node
        CHECK_EARLY_RETURN(node->left_child == nullptr
                           && node->right_child == nullptr,
                           "CommitModel: a leaf node cannot have children");
        tree[nid].set_leaf(node->info.leaf_value);
      }
    }
  }
  *out_model = std::move(model);
  return true;
}

}  // namespace builder
}  // namespace frontend
}  // namespace treelite
