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
    kEmpty, kNumericalTest, kCategoricalTest, kLeaf
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
  std::vector<uint8_t> left_categories;

  inline _Node()
    : status(_Status::kEmpty),
      parent(nullptr), left_child(nullptr), right_child(nullptr) {}
};

struct _Tree {
  _Node* root;
  std::unordered_map<int, std::shared_ptr<_Node>> nodes;
  inline _Tree() : root(nullptr), nodes() {}
};

}  // namespace anonymous

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(builder);

struct TreeBuilderImpl {
  _Tree tree;
  inline TreeBuilderImpl() : tree() {}
};

struct ModelBuilderImpl {
  std::vector<TreeBuilder> trees;
  int num_features;
  std::vector<std::pair<std::string, std::string>> cfg;
  inline ModelBuilderImpl(int num_features)
    : trees(), num_features(num_features), cfg() {
    CHECK_GT(num_features, 0) << "ModelBuilder: num_features must be positive";
  }
};

TreeBuilder::TreeBuilder()
  : pimpl(common::make_unique<TreeBuilderImpl>()), ensemble_id(nullptr) {}
TreeBuilder::~TreeBuilder() {}

bool
TreeBuilder::CreateNode(int node_key) {
  auto& nodes = pimpl->tree.nodes;
  CHECK_EARLY_RETURN(nodes.count(node_key) == 0,
                     "CreateNode: nodes with duplicate keys are not allowed");
  nodes[node_key] = common::make_unique<_Node>();
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
  CHECK_EARLY_RETURN(node->status != _Node::_Status::kLeaf,
                     "SetRootNode: cannot set a leaf node as root");
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
                                    const std::vector<uint8_t>& left_categories,
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

ModelBuilder::ModelBuilder(int num_features)
  : pimpl(common::make_unique<ModelBuilderImpl>(num_features)) {}
ModelBuilder::~ModelBuilder() {}

void
ModelBuilder::SetModelParam(const char* name, const char* value) {
  pimpl->cfg.emplace_back(name, value);
}

int
ModelBuilder::InsertTree(TreeBuilder* tree_builder, int index) {
  if (tree_builder == nullptr) {
    LOG(INFO) << "InsertTree: not a valid tree builder";
    return -1;  // fail
  }
  if (tree_builder->ensemble_id != nullptr) {
    LOG(INFO) << "InsertTree: tree is already part of another ensemble";
    return -1;  // fail
  }

  // check bounds for feature indices
  for (const auto& kv : tree_builder->pimpl->tree.nodes) {
    const int fid = static_cast<int>(kv.second->feature_id);
    if (fid < 0 || fid >= pimpl->num_features) {
      LOG(INFO) << "InsertTree: tree has an invalid split at node "
                << kv.first << ": feature id " << kv.second->feature_id
                << " is out of bound";
      return -1;  // fail
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
  model.num_features = pimpl->num_features;
  // extra parameters
  InitParamAndCheck(&model.param, pimpl->cfg);

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
        tree[nid].set_numerical_split(node->feature_id, node->info.threshold,
                                      node->default_left, node->op);
        Q.push({node->left_child, tree[nid].cleft()});
        Q.push({node->right_child, tree[nid].cright()});
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
        tree[nid].set_categorical_split(node->feature_id, node->default_left,
                                        node->left_categories);
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

}  // namespace frontend
}  // namespace treelite
