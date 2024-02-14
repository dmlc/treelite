/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file xgboost_legacy.cc
 * \brief Model loader for XGBoost model (legacy binary)
 * \author Hyunsu Cho
 */

#include <treelite/detail/file_utils.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <variant>

#include "./detail/xgboost.h"
#include "detail/string_utils.h"

namespace fs = std::filesystem;

namespace {

inline std::unique_ptr<treelite::Model> ParseStream(std::istream& fi);

}  // anonymous namespace

namespace treelite::model_loader {

std::unique_ptr<treelite::Model> LoadXGBoostModelLegacyBinary(std::string const& filename) {
  std::ifstream fi = treelite::detail::OpenFileForReadAsStream(filename);
  return ParseStream(fi);
}

std::unique_ptr<treelite::Model> LoadXGBoostModelLegacyBinary(void const* buf, std::size_t len) {
  std::istringstream fi(std::string(static_cast<char const*>(buf), len));
  return ParseStream(fi);
}

}  // namespace treelite::model_loader

/* auxiliary data structures to interpret xgboost model file */
namespace {

using bst_float = float;
using treelite::model_loader::detail::StringStartsWith;

/* peekable input stream implemented with a ring buffer */
class PeekableInputStream {
 public:
  std::size_t const MAX_PEEK_WINDOW = 1024;  // peek up to 1024 bytes

  explicit PeekableInputStream(std::istream& fi)
      : istm_(fi), buf_(MAX_PEEK_WINDOW + 1), begin_ptr_(0), end_ptr_(0) {}

  inline std::size_t Read(void* ptr, std::size_t size) {
    std::size_t const bytes_buffered = BytesBuffered();
    char* cptr = static_cast<char*>(ptr);
    if (size <= bytes_buffered) {
      // all content already buffered; consume buffer
      if (begin_ptr_ + size < MAX_PEEK_WINDOW + 1) {
        std::memcpy(cptr, &buf_[begin_ptr_], size);
        begin_ptr_ += size;
      } else {
        std::memcpy(cptr, &buf_[begin_ptr_], MAX_PEEK_WINDOW + 1 - begin_ptr_);
        std::memcpy(cptr + MAX_PEEK_WINDOW + 1 - begin_ptr_, &buf_[0],
            size + begin_ptr_ - MAX_PEEK_WINDOW - 1);
        begin_ptr_ = size + begin_ptr_ - MAX_PEEK_WINDOW - 1;
      }
      return size;
    } else {  // consume buffer entirely and read more bytes
      std::size_t const bytes_to_read = size - bytes_buffered;
      if (begin_ptr_ <= end_ptr_) {
        std::memcpy(cptr, &buf_[begin_ptr_], bytes_buffered);
      } else {
        std::memcpy(cptr, &buf_[begin_ptr_], MAX_PEEK_WINDOW + 1 - begin_ptr_);
        std::memcpy(cptr + MAX_PEEK_WINDOW + 1 - begin_ptr_, &buf_[0],
            bytes_buffered + begin_ptr_ - MAX_PEEK_WINDOW - 1);
      }
      begin_ptr_ = end_ptr_;
      istm_.read(cptr + bytes_buffered, bytes_to_read);
      return bytes_buffered + istm_.gcount();
    }
  }

  inline std::size_t PeekRead(void* ptr, std::size_t size) {
    TREELITE_CHECK_LE(size, MAX_PEEK_WINDOW)
        << "PeekableInputStream allows peeking up to " << MAX_PEEK_WINDOW << " bytes";
    char* cptr = static_cast<char*>(ptr);
    std::size_t const bytes_buffered = BytesBuffered();
    /* fill buffer with additional bytes, up to size */
    if (size > bytes_buffered) {
      std::size_t const bytes_to_read = size - bytes_buffered;
      if (end_ptr_ + bytes_to_read < MAX_PEEK_WINDOW + 1) {
        istm_.read(&buf_[end_ptr_], bytes_to_read);
        TREELITE_CHECK_EQ(istm_.gcount(), bytes_to_read) << "Failed to peek " << size << " bytes";
        end_ptr_ += bytes_to_read;
      } else {
        istm_.read(&buf_[end_ptr_], MAX_PEEK_WINDOW + 1 - end_ptr_);
        std::size_t first_read = istm_.gcount();
        istm_.read(&buf_[0], bytes_to_read + end_ptr_ - MAX_PEEK_WINDOW - 1);
        std::size_t second_read = istm_.gcount();
        TREELITE_CHECK_EQ(first_read + second_read, bytes_to_read)
            << "Ill-formed XGBoost model: Failed to peek " << size << " bytes";
        end_ptr_ = bytes_to_read + end_ptr_ - MAX_PEEK_WINDOW - 1;
      }
    }
    /* copy buffer into ptr without emptying buffer */
    if (begin_ptr_ <= end_ptr_) {  // usual case
      std::memcpy(cptr, &buf_[begin_ptr_], end_ptr_ - begin_ptr_);
    } else {  // context wrapped around the end
      std::memcpy(cptr, &buf_[begin_ptr_], MAX_PEEK_WINDOW + 1 - begin_ptr_);
      std::memcpy(cptr + MAX_PEEK_WINDOW + 1 - begin_ptr_, &buf_[0], end_ptr_);
    }

    return size;
  }

 private:
  std::istream& istm_;
  std::vector<char> buf_;
  std::size_t begin_ptr_, end_ptr_;

  inline std::size_t BytesBuffered() {
    if (begin_ptr_ <= end_ptr_) {  // usual case
      return end_ptr_ - begin_ptr_;
    } else {  // context wrapped around the end
      return MAX_PEEK_WINDOW + 1 + end_ptr_ - begin_ptr_;
    }
  }
};

template <typename T>
inline void CONSUME_BYTES(T const& fi, std::size_t size) {
  static std::vector<char> dummy(500);
  if (size > dummy.size()) {
    dummy.resize(size);
  }
  TREELITE_CHECK_EQ(fi->Read(&dummy[0], size), size)
      << "Ill-formed XGBoost model format: cannot read " << size << " bytes from the file";
}

struct LearnerModelParam {
  bst_float base_score;  // global bias
  std::uint32_t num_feature;
  std::int32_t num_class;
  std::int32_t contain_extra_attrs;
  std::int32_t contain_eval_metrics;
  std::uint32_t major_version;
  std::uint32_t minor_version;
  std::uint32_t num_target;
  std::int32_t pad2[26];
};
static_assert(sizeof(int) == sizeof(std::int32_t), "Wrong size for unsigned int");
static_assert(sizeof(unsigned) == sizeof(std::uint32_t), "Wrong size for unsigned int");
static_assert(sizeof(LearnerModelParam) == 136, "This is the size defined in XGBoost.");

struct GBTreeModelParam {
  std::int32_t num_trees;
  std::int32_t num_roots;
  std::int32_t num_feature;
  std::int32_t pad1;
  std::int64_t pad2;
  std::int32_t num_output_group;
  std::int32_t size_leaf_vector;
  std::int32_t pad3[32];
};

struct TreeParam {
  std::int32_t num_roots;
  std::int32_t num_nodes;
  std::int32_t num_deleted;
  std::int32_t max_depth;
  std::int32_t num_feature;
  std::int32_t size_leaf_vector;
  std::int32_t reserved[31];
};

struct NodeStat {
  bst_float loss_chg;
  bst_float sum_hess;
  bst_float base_weight;
  std::int32_t leaf_child_cnt;
};

class XGBTree {
 public:
  class Node {
   public:
    Node() : sindex_(0) {
      // assert compact alignment
      static_assert(sizeof(Node) == 4 * sizeof(int) + sizeof(Info), "Node: 64 bit align");
    }
    inline int cleft() const {
      return this->cleft_;
    }
    inline int cright() const {
      return this->cright_;
    }
    inline int cdefault() const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    inline unsigned split_index() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    inline bool default_left() const {
      return (sindex_ >> 31) != 0;
    }
    inline bool is_leaf() const {
      return cleft_ == -1;
    }
    inline bst_float leaf_value() const {
      return (this->info_).leaf_value;
    }
    inline bst_float split_cond() const {
      return (this->info_).split_cond;
    }
    inline int parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    inline bool is_root() const {
      return parent_ == -1;
    }
    inline void set_leaf(bst_float value) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = -1;
    }

    union Info {
      bst_float leaf_value;
      bst_float split_cond;
    };
    std::int32_t parent_;
    std::int32_t cleft_, cright_;
    std::uint32_t sindex_;
    Info info_;

    inline bool is_deleted() const {
      return sindex_ == std::numeric_limits<std::uint32_t>::max();
    }
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) {
        pidx |= (1U << 31);
      }
      this->parent_ = pidx;
    }
  };

  TreeParam param;
  std::vector<Node> nodes;
  std::vector<NodeStat> stats;

 public:
  /*! \brief get node given nid */
  inline Node& operator[](int nid) {
    return nodes[nid];
  }
  /*! \brief get node given nid */
  inline Node const& operator[](int nid) const {
    return nodes[nid];
  }
  /*! \brief get node statistics given nid */
  inline NodeStat& Stat(int nid) {
    return stats[nid];
  }
  /*! \brief get node statistics given nid */
  inline NodeStat const& Stat(int nid) const {
    return stats[nid];
  }
  inline void Init() {
    param.num_nodes = 1;
    nodes.resize(1);
    nodes[0].set_leaf(0.0f);
    nodes[0].set_parent(-1);
  }
  inline void Load(PeekableInputStream* fi, LearnerModelParam const& mparam) {
    TREELITE_CHECK_EQ(fi->Read(&param, sizeof(TreeParam)), sizeof(TreeParam))
        << "Ill-formed XGBoost model file: can't read TreeParam";
    TREELITE_CHECK_GT(param.num_nodes, 0) << "Ill-formed XGBoost model file: a tree can't be empty";
    nodes.resize(param.num_nodes);
    stats.resize(param.num_nodes);
    TREELITE_CHECK_EQ(
        fi->Read(nodes.data(), sizeof(Node) * nodes.size()), sizeof(Node) * nodes.size())
        << "Ill-formed XGBoost model file: cannot read specified number of nodes";
    TREELITE_CHECK_EQ(
        fi->Read(stats.data(), sizeof(NodeStat) * stats.size()), sizeof(NodeStat) * stats.size())
        << "Ill-formed XGBoost model file: cannot read specified number of nodes";
    if (param.size_leaf_vector != 0 && mparam.major_version < 2) {
      std::uint64_t len;
      TREELITE_CHECK_EQ(fi->Read(&len, sizeof(len)), sizeof(len))
          << "Ill-formed XGBoost model file";
      if (len > 0) {
        CONSUME_BYTES(fi, sizeof(bst_float) * len);
      }
    } else if (mparam.major_version == 2) {
      TREELITE_CHECK_EQ(param.size_leaf_vector, 1)
          << "Multi-target models are not supported with binary serialization. "
          << "Please save the XGBoost model using the JSON format.";
    }
    TREELITE_CHECK_EQ(param.num_roots, 1)
        << "Invalid XGBoost model file: treelite does not support trees "
        << "with multiple roots";
  }
};

inline std::unique_ptr<treelite::Model> ParseStream(std::istream& fi) {
  std::vector<XGBTree> xgb_trees_;
  LearnerModelParam mparam_;  // model parameter
  GBTreeModelParam gbm_param_;  // GBTree training parameter
  std::string name_gbm_;
  std::string name_obj_;

  /* 1. Parse input stream */
  std::unique_ptr<PeekableInputStream> fp(new PeekableInputStream(fi));
  // backward compatible header check.
  std::string header;
  header.resize(4);
  if (fp->PeekRead(&header[0], 4) == 4) {
    TREELITE_CHECK_NE(header, "bs64")
        << "Ill-formed XGBoost model file: Base64 format no longer supported";
    if (header == "binf") {
      CONSUME_BYTES(fp, 4);
    }
  }
  // read parameter
  TREELITE_CHECK_EQ(fp->Read(&mparam_, sizeof(mparam_)), sizeof(mparam_))
      << "Ill-formed XGBoost model file: corrupted header";
  {
    std::uint64_t len;
    TREELITE_CHECK_EQ(fp->Read(&len, sizeof(len)), sizeof(len))
        << "Ill-formed XGBoost model file: corrupted header";
    if (len != 0) {
      name_obj_.resize(len);
      TREELITE_CHECK_EQ(fp->Read(&name_obj_[0], len), len)
          << "Ill-formed XGBoost model file: corrupted header";
    }
  }

  {
    std::uint64_t len;
    TREELITE_CHECK_EQ(fp->Read(&len, sizeof(len)), sizeof(len))
        << "Ill-formed XGBoost model file: corrupted header";
    name_gbm_.resize(len);
    if (len > 0) {
      TREELITE_CHECK_EQ(fp->Read(&name_gbm_[0], len), len)
          << "Ill-formed XGBoost model file: corrupted header";
    }
  }

  /* loading GBTree */
  TREELITE_CHECK(name_gbm_ == "gbtree" || name_gbm_ == "dart")
      << "Invalid XGBoost model file: "
      << "Gradient booster must be gbtree or dart type.";

  TREELITE_CHECK_EQ(fp->Read(&gbm_param_, sizeof(gbm_param_)), sizeof(gbm_param_))
      << "Invalid XGBoost model file: corrupted GBTree parameters";
  TREELITE_CHECK_GE(gbm_param_.num_trees, 0)
      << "Invalid XGBoost model file: num_trees must be 0 or greater";
  for (int i = 0; i < gbm_param_.num_trees; ++i) {
    xgb_trees_.emplace_back();
    xgb_trees_.back().Load(fp.get(), mparam_);
  }
  if (mparam_.major_version < 1 || (mparam_.major_version == 1 && mparam_.minor_version < 6)) {
    // In XGBoost 1.6, num_roots is used as num_parallel_tree, so don't check
    TREELITE_CHECK_EQ(gbm_param_.num_roots, 1) << "multi-root trees not supported";
  }
  std::vector<std::int32_t> tree_info;
  tree_info.resize(gbm_param_.num_trees);
  if (gbm_param_.num_trees > 0) {
    TREELITE_CHECK_EQ(fp->Read(tree_info.data(), sizeof(std::int32_t) * tree_info.size()),
        sizeof(std::int32_t) * tree_info.size());
  }
  // Load weight drop values (per tree) for dart models.
  std::vector<bst_float> weight_drop;
  if (name_gbm_ == "dart") {
    weight_drop.resize(gbm_param_.num_trees);
    std::uint64_t sz;
    fi.read(reinterpret_cast<char*>(&sz), sizeof(std::uint64_t));
    TREELITE_CHECK_EQ(sz, gbm_param_.num_trees);
    if (gbm_param_.num_trees != 0) {
      for (std::uint64_t i = 0; i < sz; ++i) {
        fi.read(reinterpret_cast<char*>(&weight_drop[i]), sizeof(bst_float));
      }
    }
  }

  /* 2. Set metadata */
  auto const num_feature = static_cast<std::int32_t>(mparam_.num_feature);
  bool const average_tree_output = false;

  // XGBoost binary format only supports decision trees with scalar outputs
  auto num_target = static_cast<std::int32_t>(mparam_.num_target);
  TREELITE_CHECK_GE(num_target, 0) << "num_target too big and caused an integer overflow";
  if (num_target == 0) {
    num_target = 1;
  }
  TREELITE_CHECK_GE(num_target, 1) << "num_target must be at least 1";
  std::int32_t const num_class = std::max(mparam_.num_class, static_cast<std::int32_t>(1));
  std::array<std::int32_t, 2> const leaf_vector_shape{1, 1};
  auto const num_tree = static_cast<std::int32_t>(gbm_param_.num_trees);

  // Assume: Either num_target or num_class must be 1
  TREELITE_CHECK(num_target == 1 || num_class == 1);

  treelite::TaskType task_type;
  std::vector<std::int32_t> target_id, class_id;
  if (num_class > 1) {
    // Multi-class classifier with grove per class
    // i-th tree produces output for class (i % num_class)
    // Note: num_parallel_tree can change this behavior, so it's best to go with
    // tree_info field provided by XGBoost
    task_type = treelite::TaskType::kMultiClf;
    class_id = std::vector<std::int32_t>(num_tree);
    for (std::int32_t tree_id = 0; tree_id < num_tree; ++tree_id) {
      class_id[tree_id] = tree_info[tree_id];
    }
    target_id = std::vector<std::int32_t>(num_tree, 0);
  } else {
    // binary classifier or regressor
    if (StringStartsWith(name_obj_, "binary:")) {
      task_type = treelite::TaskType::kBinaryClf;
    } else if (StringStartsWith(name_obj_, "rank:")) {
      task_type = treelite::TaskType::kLearningToRank;
    } else {
      task_type = treelite::TaskType::kRegressor;
    }
    class_id = std::vector<std::int32_t>(num_tree, 0);
    target_id = std::vector<std::int32_t>(num_tree);
    for (std::int32_t tree_id = 0; tree_id < num_tree; ++tree_id) {
      target_id[tree_id] = tree_info[tree_id];
    }
  }

  treelite::model_builder::PostProcessorFunc postprocessor{
      treelite::model_loader::detail::xgboost::GetPostProcessor(name_obj_)};
  treelite::model_builder::Metadata metadata{num_feature, task_type, average_tree_output,
      num_target, std::vector<std::int32_t>(num_target, num_class), leaf_vector_shape};
  treelite::model_builder::TreeAnnotation tree_annotation{num_tree, target_id, class_id};

  // Set base scores. For now, XGBoost only supports a scalar base score for all targets / classes.
  auto base_score = static_cast<double>(mparam_.base_score);
  // Before XGBoost 1.0.0, the global bias saved in model is a transformed value.  After
  // 1.0 it's the original value provided by user.
  bool const need_transform_to_margin = mparam_.major_version >= 1;
  if (need_transform_to_margin) {
    base_score = treelite::model_loader::detail::xgboost::TransformBaseScoreToMargin(
        postprocessor.name, base_score);
  }
  std::size_t const len_base_scores = num_target * num_class;
  std::vector<double> base_scores(len_base_scores, base_score);

  auto builder = treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
      treelite::TypeInfo::kFloat32, metadata, tree_annotation, postprocessor, base_scores);

  /* 3. Build trees */
  for (int tree_id = 0; tree_id < xgb_trees_.size(); ++tree_id) {
    auto const& xgb_tree = xgb_trees_[tree_id];
    builder->StartTree();
    for (int node_id = 0; node_id < xgb_tree.nodes.size(); ++node_id) {
      auto const& node = xgb_tree.nodes[node_id];
      if (!node.is_deleted()) {
        builder->StartNode(node_id);
        NodeStat const stat = xgb_tree.stats[node_id];
        if (node.is_leaf()) {
          bst_float leaf_value = node.leaf_value();
          // Fold weight drop into leaf value for dart models.
          if (!weight_drop.empty()) {
            leaf_value *= weight_drop[tree_id];
          }
          builder->LeafScalar(leaf_value);
        } else {
          bst_float const split_cond = node.split_cond();
          builder->NumericalTest(static_cast<std::int32_t>(node.split_index()),
              static_cast<float>(split_cond), node.default_left(), treelite::Operator::kLT,
              node.cleft(), node.cright());
          builder->Gain(stat.loss_chg);
        }
        builder->SumHess(stat.sum_hess);
        builder->EndNode();
      }
    }
    builder->EndTree();
  }
  return builder->CommitModel();
}

}  // anonymous namespace
