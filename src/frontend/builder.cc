/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file builder.cc
 * \brief model builder frontend
 * \author Hyunsu Cho
 */

#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <treelite/logging.h>
#include <memory>
#include <queue>

/* data structures with underscore prefixes are internal use only and don't have external linkage */
namespace {

struct NodeDraft {
  enum class Status : int8_t {
    kEmpty, kNumericalTest, kCategoricalTest, kLeaf
  };
  /*
   * leaf vector: only used for random forests with multi-class classification
   */
  std::vector<treelite::frontend::Value> leaf_vector;
  Status status;
  /* pointers to parent, left and right children */
  NodeDraft* parent;
  NodeDraft* left_child;
  NodeDraft* right_child;
  // split feature index
  unsigned feature_id;
  // default direction for missing values
  bool default_left;
  // leaf value (only for leaf nodes)
  treelite::frontend::Value leaf_value;
  // threshold (only for non-leaf nodes)
  treelite::frontend::Value threshold;
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

  inline NodeDraft()
    : status(Status::kEmpty), parent(nullptr), left_child(nullptr), right_child(nullptr) {}
};

struct TreeDraft {
  NodeDraft* root;
  std::unordered_map<int, std::unique_ptr<NodeDraft>> nodes;
  treelite::TypeInfo threshold_type;
  treelite::TypeInfo leaf_output_type;
  inline TreeDraft(treelite::TypeInfo threshold_type, treelite::TypeInfo leaf_output_type)
    : root(nullptr), nodes(), threshold_type(threshold_type), leaf_output_type(leaf_output_type) {}
};

}  // anonymous namespace

namespace treelite {
namespace frontend {

struct TreeBuilderImpl {
  TreeDraft tree;
  inline TreeBuilderImpl(TypeInfo threshold_type, TypeInfo leaf_output_type)
    : tree(threshold_type, leaf_output_type) {}
};

struct ModelBuilderImpl {
  std::vector<TreeBuilder> trees;
  int num_feature;
  int num_class;
  bool average_tree_output;
  TypeInfo threshold_type;
  TypeInfo leaf_output_type;
  std::vector<std::pair<std::string, std::string>> cfg;
  inline ModelBuilderImpl(int num_feature, int num_class, bool average_tree_output,
                          TypeInfo threshold_type, TypeInfo leaf_output_type)
    : trees(), num_feature(num_feature), num_class(num_class),
      average_tree_output(average_tree_output), threshold_type(threshold_type),
      leaf_output_type(leaf_output_type), cfg() {
    TREELITE_CHECK_GT(num_feature, 0) << "ModelBuilder: num_feature must be positive";
    TREELITE_CHECK_GT(num_class, 0) << "ModelBuilder: num_class must be positive";
    TREELITE_CHECK(threshold_type != TypeInfo::kInvalid)
      << "ModelBuilder: threshold_type can't be invalid";
    TREELITE_CHECK(leaf_output_type != TypeInfo::kInvalid)
      << "ModelBuilder: leaf_output_type can't be invalid";
  }
  // Templatized implementation of CommitModel()
  template <typename ThresholdType, typename LeafOutputType>
  void CommitModelImpl(ModelImpl<ThresholdType, LeafOutputType>* out_model);
};

template <typename ThresholdType, typename LeafOutputType>
void SetLeafVector(Tree<ThresholdType, LeafOutputType>* tree, int nid,
                   const std::vector<Value>& leaf_vector) {
  const size_t leaf_vector_size = leaf_vector.size();
  const TypeInfo expected_leaf_type = TypeToInfo<LeafOutputType>();
  std::vector<LeafOutputType> out_leaf_vector;
  for (size_t i = 0; i < leaf_vector_size; ++i) {
    const Value& leaf_value = leaf_vector[i];
    TREELITE_CHECK(leaf_value.GetValueType() == expected_leaf_type)
      << "Leaf value at index " << i << " has incorrect type. Expected: "
      << TypeInfoToString(expected_leaf_type) << ", Given: "
      << TypeInfoToString(leaf_value.GetValueType());
    out_leaf_vector.push_back(leaf_value.Get<LeafOutputType>());
  }
  tree->SetLeafVector(nid, out_leaf_vector);
}

Value::Value() : handle_(nullptr), type_(TypeInfo::kInvalid) {}

template <typename T>
Value
Value::Create(T init_value) {
  Value value;
  std::unique_ptr<T> ptr = std::make_unique<T>(init_value);
  value.handle_.reset(ptr.release());
  value.type_ = TypeToInfo<T>();
  return value;
}

template <typename ValueType>
class CreateHandle {
 public:
  inline static std::shared_ptr<void> Dispatch(const void* init_value) {
    const auto* v_ptr = static_cast<const ValueType*>(init_value);
    TREELITE_CHECK(v_ptr);
    ValueType v = *v_ptr;
    return std::make_shared<ValueType>(v);
  }
};

Value
Value::Create(const void* init_value, TypeInfo type) {
  Value value;
  TREELITE_CHECK(type != TypeInfo::kInvalid) << "Type must be valid";
  value.type_ = type;
  value.handle_ = DispatchWithTypeInfo<CreateHandle>(type, init_value);
  return value;
}

template <typename T>
T&
Value::Get() {
  TREELITE_CHECK(handle_);
  T* out = static_cast<T*>(handle_.get());
  TREELITE_CHECK(out);
  return *out;
}

template <typename T>
const T&
Value::Get() const {
  TREELITE_CHECK(handle_);
  const T* out = static_cast<const T*>(handle_.get());
  TREELITE_CHECK(out);
  return *out;
}

TypeInfo
Value::GetValueType() const {
  return type_;
}

TreeBuilder::TreeBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type)
  : pimpl_(new TreeBuilderImpl(threshold_type, leaf_output_type)), ensemble_id_(nullptr) {}
TreeBuilder::~TreeBuilder() = default;

void
TreeBuilder::CreateNode(int node_key) {
  auto& nodes = pimpl_->tree.nodes;
  TREELITE_CHECK_EQ(nodes.count(node_key), 0)
    << "CreateNode: nodes with duplicate keys are not allowed";
  nodes[node_key] = std::make_unique<NodeDraft>();
}

void
TreeBuilder::DeleteNode(int node_key) {
  auto& tree = pimpl_->tree;
  auto& nodes = tree.nodes;
  TREELITE_CHECK_GT(nodes.count(node_key), 0) << "DeleteNode: no node found with node_key";
  NodeDraft* node = nodes[node_key].get();
  if (tree.root == node) {  // deleting root
    tree.root = nullptr;
  }
  if (node->left_child != nullptr) {  // deleting a parent
    node->left_child->parent = nullptr;
  }
  if (node->right_child != nullptr) {  // deleting a parent
    node->right_child->parent = nullptr;
  }
  if (node == tree.root) {  // deleting root
    tree.root = nullptr;
    nodes.clear();
  } else {
    nodes.erase(node_key);
  }
}

void
TreeBuilder::SetRootNode(int node_key) {
  auto& tree = pimpl_->tree;
  auto& nodes = tree.nodes;
  TREELITE_CHECK_GT(nodes.count(node_key), 0) << "SetRootNode: no node found with node_key";
  NodeDraft* node = nodes[node_key].get();
  TREELITE_CHECK(!node->parent) << "SetRootNode: a root node cannot have a parent";
  tree.root = node;
}

void
TreeBuilder::SetNumericalTestNode(int node_key, unsigned feature_id, const char* opname,
                                  Value threshold, bool default_left, int left_child_key,
                                  int right_child_key) {
  Operator op = LookupOperatorByName(opname);
  SetNumericalTestNode(node_key, feature_id, op, std::move(threshold), default_left,
                       left_child_key, right_child_key);
}

void
TreeBuilder::SetNumericalTestNode(int node_key, unsigned feature_id, Operator op, Value threshold,
                                  bool default_left, int left_child_key, int right_child_key) {
  auto& tree = pimpl_->tree;
  auto& nodes = tree.nodes;
  TREELITE_CHECK(tree.threshold_type == threshold.GetValueType())
    << "SetNumericalTestNode: threshold has an incorrect type. "
    << "Expected: " << TypeInfoToString(tree.threshold_type)
    << ", Given: " << TypeInfoToString(threshold.GetValueType());
  TREELITE_CHECK_GT(nodes.count(node_key), 0)
    << "SetNumericalTestNode: no node found with node_key";
  TREELITE_CHECK_GT(nodes.count(left_child_key), 0)
    << "SetNumericalTestNode: no node found with left_child_key";
  TREELITE_CHECK_GT(nodes.count(right_child_key), 0)
    << "SetNumericalTestNode: no node found with right_child_key";
  NodeDraft* node = nodes[node_key].get();
  NodeDraft* left_child = nodes[left_child_key].get();
  NodeDraft* right_child = nodes[right_child_key].get();
  TREELITE_CHECK(node->status == NodeDraft::Status::kEmpty)
    << "SetNumericalTestNode: cannot modify a non-empty node";
  TREELITE_CHECK(!left_child->parent)
    << "SetNumericalTestNode: node designated as left child already has a parent";
  TREELITE_CHECK(!right_child->parent)
    << "SetNumericalTestNode: node designated as right child already has a parent";
  TREELITE_CHECK(left_child != tree.root && right_child != tree.root)
    << "SetNumericalTestNode: the root node cannot be a child";
  node->status = NodeDraft::Status::kNumericalTest;
  node->left_child = nodes[left_child_key].get();
  node->left_child->parent = node;
  node->right_child = nodes[right_child_key].get();
  node->right_child->parent = node;
  node->feature_id = feature_id;
  node->default_left = default_left;
  node->threshold = std::move(threshold);
  node->op = op;
}

void
TreeBuilder::SetCategoricalTestNode(int node_key, unsigned feature_id,
                                    const std::vector<uint32_t>& left_categories, bool default_left,
                                    int left_child_key, int right_child_key) {
  auto &tree = pimpl_->tree;
  auto &nodes = tree.nodes;
  TREELITE_CHECK_GT(nodes.count(node_key), 0)
    << "SetCategoricalTestNode: no node found with node_key";
  TREELITE_CHECK_GT(nodes.count(left_child_key), 0)
    << "SetCategoricalTestNode: no node found with left_child_key";
  TREELITE_CHECK_GT(nodes.count(right_child_key), 0)
    << "SetCategoricalTestNode: no node found with right_child_key";
  NodeDraft* node = nodes[node_key].get();
  NodeDraft* left_child = nodes[left_child_key].get();
  NodeDraft* right_child = nodes[right_child_key].get();
  TREELITE_CHECK(node->status == NodeDraft::Status::kEmpty)
    << "SetCategoricalTestNode: cannot modify a non-empty node";
  TREELITE_CHECK(!left_child->parent)
    << "SetCategoricalTestNode: node designated as left child already has a parent";
  TREELITE_CHECK(!right_child->parent)
    << "SetCategoricalTestNode: node designated as right child already has a parent";
  TREELITE_CHECK(left_child != tree.root && right_child != tree.root)
    << "SetCategoricalTestNode: the root node cannot be a child";
  node->status = NodeDraft::Status::kCategoricalTest;
  node->left_child = nodes[left_child_key].get();
  node->left_child->parent = node;
  node->right_child = nodes[right_child_key].get();
  node->right_child->parent = node;
  node->feature_id = feature_id;
  node->default_left = default_left;
  node->left_categories = left_categories;
}

void
TreeBuilder::SetLeafNode(int node_key, Value leaf_value) {
  auto& tree = pimpl_->tree;
  auto& nodes = tree.nodes;
  TREELITE_CHECK(tree.leaf_output_type == leaf_value.GetValueType())
    << "SetLeafNode: leaf_value has an incorrect type. "
    << "Expected: " << TypeInfoToString(tree.leaf_output_type)
    << ", Given: " << TypeInfoToString(leaf_value.GetValueType());
  TREELITE_CHECK_GT(nodes.count(node_key), 0) << "SetLeafNode: no node found with node_key";
  NodeDraft* node = nodes[node_key].get();
  TREELITE_CHECK(node->status == NodeDraft::Status::kEmpty)
    << "SetLeafNode: cannot modify a non-empty node";
  node->status = NodeDraft::Status::kLeaf;
  node->leaf_value = std::move(leaf_value);
}

void
TreeBuilder::SetLeafVectorNode(int node_key, const std::vector<Value>& leaf_vector) {
  auto& tree = pimpl_->tree;
  auto& nodes = tree.nodes;
  const size_t leaf_vector_len = leaf_vector.size();
  for (size_t i = 0; i < leaf_vector_len; ++i) {
    const Value& leaf_value = leaf_vector[i];
    TREELITE_CHECK(tree.leaf_output_type == leaf_value.GetValueType())
      << "SetLeafVectorNode: the element " << i << " in leaf_vector has an incorrect type. "
      << "Expected: " << TypeInfoToString(tree.leaf_output_type)
      << ", Given: " << TypeInfoToString(leaf_value.GetValueType());
  }
  TREELITE_CHECK_GT(nodes.count(node_key), 0)
    << "SetLeafVectorNode: no node found with node_key";
  NodeDraft* node = nodes[node_key].get();
  TREELITE_CHECK(node->status == NodeDraft::Status::kEmpty)
    << "SetLeafVectorNode: cannot modify a non-empty node";
  node->status = NodeDraft::Status::kLeaf;
  node->leaf_vector = leaf_vector;
}

ModelBuilder::ModelBuilder(int num_feature, int num_class, bool average_tree_output,
                           TypeInfo threshold_type, TypeInfo leaf_output_type)
  : pimpl_(new ModelBuilderImpl(num_feature, num_class, average_tree_output,
                                threshold_type, leaf_output_type)) {}
ModelBuilder::~ModelBuilder() = default;

void
ModelBuilder::SetModelParam(const char* name, const char* value) {
  pimpl_->cfg.emplace_back(name, value);
}

int
ModelBuilder::InsertTree(TreeBuilder* tree_builder, int index) {
  if (tree_builder == nullptr) {
    TREELITE_LOG(FATAL) << "InsertTree: not a valid tree builder";
    return -1;
  }
  if (tree_builder->ensemble_id_ != nullptr) {
    TREELITE_LOG(FATAL) << "InsertTree: tree is already part of another ensemble";
    return -1;
  }
  if (tree_builder->pimpl_->tree.threshold_type != this->pimpl_->threshold_type) {
    TREELITE_LOG(FATAL)
      << "InsertTree: cannot insert the tree into the ensemble, because the ensemble requires all "
      << "member trees to use " << TypeInfoToString(this->pimpl_->threshold_type)
      << " type for split thresholds whereas the tree is using "
      << TypeInfoToString(tree_builder->pimpl_->tree.threshold_type);
    return -1;
  }
  if (tree_builder->pimpl_->tree.leaf_output_type != this->pimpl_->leaf_output_type) {
    TREELITE_LOG(FATAL)
      << "InsertTree: cannot insert the tree into the ensemble, because the ensemble requires all "
      << "member trees to use " << TypeInfoToString(this->pimpl_->leaf_output_type)
      << " type for leaf outputs whereas the tree is using "
      << TypeInfoToString(tree_builder->pimpl_->tree.leaf_output_type);
    return -1;
  }

  // check bounds for feature indices
  for (const auto& kv : tree_builder->pimpl_->tree.nodes) {
    const NodeDraft::Status status = kv.second->status;
    if (status == NodeDraft::Status::kNumericalTest ||
        status == NodeDraft::Status::kCategoricalTest) {
      const int fid = static_cast<int>(kv.second->feature_id);
      if (fid < 0 || fid >= this->pimpl_->num_feature) {
        TREELITE_LOG(FATAL) << "InsertTree: tree has an invalid split at node "
                            << kv.first << ": feature id "
                            << kv.second->feature_id << " is out of bound";
        return -1;
      }
    }
  }

  // perform insertion
  auto& trees = pimpl_->trees;
  if (index == -1) {
    trees.push_back(std::move(*tree_builder));
    tree_builder->ensemble_id_ = this;
    return static_cast<int>(trees.size());
  } else {
    if (static_cast<size_t>(index) <= trees.size()) {
      trees.insert(trees.begin() + index, std::move(*tree_builder));
      tree_builder->ensemble_id_ = this;
      return index;
    } else {
      TREELITE_LOG(FATAL) << "InsertTree: index out of bound";
      return -1;
    }
  }
}

TreeBuilder*
ModelBuilder::GetTree(int index) {
  return &pimpl_->trees.at(index);
}

const TreeBuilder*
ModelBuilder::GetTree(int index) const {
  return &pimpl_->trees.at(index);
}

void
ModelBuilder::DeleteTree(int index) {
  auto& trees = pimpl_->trees;
  TREELITE_CHECK_LT(static_cast<size_t>(index), trees.size())
    << "DeleteTree: index out of bound";
  trees.erase(trees.begin() + index);
}

std::unique_ptr<Model>
ModelBuilder::CommitModel() {
  std::unique_ptr<Model> model_ptr = Model::Create(pimpl_->threshold_type,
                                                   pimpl_->leaf_output_type);
  model_ptr->Dispatch([this](auto& model) {
    this->pimpl_->CommitModelImpl(&model);
  });
  return model_ptr;
}

template <typename ThresholdType, typename LeafOutputType>
void
ModelBuilderImpl::CommitModelImpl(ModelImpl<ThresholdType, LeafOutputType>* out_model) {
  ModelImpl<ThresholdType, LeafOutputType>& model = *out_model;
  model.num_feature = this->num_feature;
  model.average_tree_output = this->average_tree_output;
  model.task_param.output_type = TaskParam::OutputType::kFloat;
  model.task_param.num_class = this->num_class;
  // extra parameters
  InitParamAndCheck(&model.param, this->cfg);

  // flag to check consistent use of leaf vector
  // 0: no leaf should use leaf vector
  // 1: every leaf should use leaf vector
  // -1: indeterminate
  int8_t flag_leaf_vector = -1;

  for (const auto& tree_builder : this->trees) {
    const auto& _tree = tree_builder.pimpl_->tree;
    TREELITE_CHECK(_tree.root) << "CommitModel: a tree has no root node";
    TREELITE_CHECK(_tree.root->status != NodeDraft::Status::kEmpty)
      << "SetRootNode: cannot set an empty node as root";
    model.trees.emplace_back();
    Tree<ThresholdType, LeafOutputType>& tree = model.trees.back();
    tree.Init();

    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<const NodeDraft*, int>> Q;  // (internal pointer, ID)
    Q.push({_tree.root, 0});  // assign 0 to root
    while (!Q.empty()) {
      const NodeDraft* node;
      int nid;
      std::tie(node, nid) = Q.front();
      Q.pop();
      TREELITE_CHECK(node->status != NodeDraft::Status::kEmpty)
        << "CommitModel: encountered an empty node in the middle of a tree";
      if (node->status == NodeDraft::Status::kNumericalTest) {
        TREELITE_CHECK(node->left_child) << "CommitModel: a test node lacks a left child";
        TREELITE_CHECK(node->right_child) << "CommitModel: a test node lacks a right child";
        TREELITE_CHECK(node->left_child->parent == node)
          << "CommitModel: left child has wrong parent";
        TREELITE_CHECK(node->right_child->parent == node)
          << "CommitModel: right child has wrong parent";
        tree.AddChilds(nid);
        TREELITE_CHECK(node->threshold.GetValueType() == TypeToInfo<ThresholdType>())
          << "CommitModel: The specified threshold has incorrect type. Expected: "
          << TypeInfoToString(TypeToInfo<ThresholdType>())
          << " Given: " << TypeInfoToString(node->threshold.GetValueType());
        ThresholdType threshold = node->threshold.Get<ThresholdType>();
        tree.SetNumericalSplit(nid, node->feature_id, threshold, node->default_left, node->op);
        Q.push({node->left_child, tree.LeftChild(nid)});
        Q.push({node->right_child, tree.RightChild(nid)});
      } else if (node->status == NodeDraft::Status::kCategoricalTest) {
        TREELITE_CHECK(node->left_child) << "CommitModel: a test node lacks a left child";
        TREELITE_CHECK(node->right_child) << "CommitModel: a test node lacks a right child";
        TREELITE_CHECK(node->left_child->parent == node)
          << "CommitModel: left child has wrong parent";
        TREELITE_CHECK(node->right_child->parent == node)
          << "CommitModel: right child has wrong parent";
        tree.AddChilds(nid);
        tree.SetCategoricalSplit(nid, node->feature_id, node->default_left, node->left_categories,
                                 false);
        Q.push({node->left_child, tree.LeftChild(nid)});
        Q.push({node->right_child, tree.RightChild(nid)});
      } else {  // leaf node
        TREELITE_CHECK(node->left_child == nullptr && node->right_child == nullptr)
          << "CommitModel: a leaf node cannot have children";
        if (!node->leaf_vector.empty()) {  // leaf vector exists
          TREELITE_CHECK_NE(flag_leaf_vector, 0)
            << "CommitModel: Inconsistent use of leaf vector: if one leaf node uses a leaf vector, "
            << "*every* leaf node must use a leaf vector";
          flag_leaf_vector = 1;  // now every leaf must use leaf vector
          TREELITE_CHECK_EQ(node->leaf_vector.size(), model.task_param.num_class)
            << "CommitModel: The length of leaf vector must be identical to the number of output "
            << "groups";
          SetLeafVector(&tree, nid, node->leaf_vector);
        } else {  // ordinary leaf
          TREELITE_CHECK_NE(flag_leaf_vector, 1)
            << "CommitModel: Inconsistent use of leaf vector: if one leaf node does not use a leaf "
            << "vector, *no other* leaf node can use a leaf vector";
          flag_leaf_vector = 0;  // now no leaf can use leaf vector
          TREELITE_CHECK(node->leaf_value.GetValueType() == TypeToInfo<LeafOutputType>())
            << "CommitModel: The specified leaf value has incorrect type. Expected: "
            << TypeInfoToString(TypeToInfo<LeafOutputType>())
            << " Given: " << TypeInfoToString(node->leaf_value.GetValueType());
          LeafOutputType leaf_value = node->leaf_value.Get<LeafOutputType>();
          tree.SetLeaf(nid, leaf_value);
        }
      }
    }
  }
  if (flag_leaf_vector == 0) {
    model.task_param.leaf_vector_size = 1;
    if (model.task_param.num_class > 1) {
      // multi-class classifier, XGBoost/LightGBM style
      model.task_type = TaskType::kMultiClfGrovePerClass;
      model.task_param.grove_per_class = true;
      TREELITE_CHECK_EQ(this->trees.size() % model.task_param.num_class, 0)
        << "For multi-class classifiers with gradient boosted trees, the number of trees must be "
        << "evenly divisible by the number of output groups";
    } else {
      // binary classifier or regressor
      model.task_type = TaskType::kBinaryClfRegr;
      model.task_param.grove_per_class = false;
    }
  } else if (flag_leaf_vector == 1) {
    // multi-class classifier, sklearn RF style
    model.task_type = TaskType::kMultiClfProbDistLeaf;
    model.task_param.grove_per_class = false;
    TREELITE_CHECK_GT(model.task_param.num_class, 1)
      << "Expected leaf vectors with length exceeding 1";
    model.task_param.leaf_vector_size = model.task_param.num_class;
  } else {
    TREELITE_LOG(FATAL) << "Impossible thing happened: model has no leaf node!";
  }
}

template Value Value::Create(uint32_t init_value);
template Value Value::Create(float init_value);
template Value Value::Create(double init_value);

}  // namespace frontend
}  // namespace treelite
