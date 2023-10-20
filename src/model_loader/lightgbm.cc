/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file lightgbm.cc
 * \brief Model loader for LightGBM model
 * \author Hyunsu Cho
 */

#include "./detail/lightgbm.h"

#include <treelite/detail/file_utils.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "./detail/string_utils.h"

namespace {

inline std::unique_ptr<treelite::Model> ParseStream(std::istream& fi);

}  // anonymous namespace

namespace treelite::model_loader {

std::unique_ptr<treelite::Model> LoadLightGBMModel(std::string const& filename) {
  std::ifstream fi = treelite::detail::OpenFileForReadAsStream(filename);
  return ParseStream(fi);
}

std::unique_ptr<treelite::Model> LoadLightGBMModelFromString(char const* model_str) {
  std::istringstream is(model_str);
  return ParseStream(is);
}

}  // namespace treelite::model_loader

/* auxiliary data structures to interpret lightgbm model file */
namespace {

template <typename T>
inline T TextToNumber(std::string const& str) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value
                    || std::is_same<T, int>::value || std::is_same<T, std::int8_t>::value
                    || std::is_same<T, std::uint32_t>::value
                    || std::is_same<T, std::uint64_t>::value,
      "unsupported data type for TextToNumber; use float, double, "
      "int, int8_t, uint32_t, or uint64_t");
}

template <>
inline float TextToNumber(std::string const& str) {
  errno = 0;
  char* endptr;
  float val = std::strtof(str.c_str(), &endptr);
  if (errno == ERANGE) {
    TREELITE_LOG(FATAL) << "Range error while converting string to double";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid floating-point number";
  }
  return val;
}

template <>
inline double TextToNumber(std::string const& str) {
  errno = 0;
  char* endptr;
  double val = std::strtod(str.c_str(), &endptr);
  if (errno == ERANGE) {
    TREELITE_LOG(FATAL) << "Range error while converting string to double";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid floating-point number";
  }
  return val;
}

template <>
inline int TextToNumber(std::string const& str) {
  errno = 0;
  char* endptr;
  auto val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < std::numeric_limits<int>::min()
      || val > std::numeric_limits<int>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to int";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int>(val);
}

template <>
inline std::int8_t TextToNumber(std::string const& str) {
  errno = 0;
  char* endptr;
  auto val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < std::numeric_limits<std::int8_t>::min()
      || val > std::numeric_limits<std::int8_t>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to int8_t";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<std::int8_t>(val);
}

template <>
inline std::uint32_t TextToNumber(std::string const& str) {
  errno = 0;
  char* endptr;
  auto val = std::strtoul(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val > std::numeric_limits<std::uint32_t>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to uint32_t";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<std::uint32_t>(val);
}

template <>
inline std::uint64_t TextToNumber(std::string const& str) {
  errno = 0;
  char* endptr;
  auto val = std::strtoull(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val > std::numeric_limits<std::uint64_t>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to uint64_t";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<std::uint64_t>(val);
}

inline std::vector<std::string> Split(std::string const& text, char delim) {
  std::vector<std::string> array;
  std::istringstream ss(text);
  std::string token;
  while (std::getline(ss, token, delim)) {
    array.push_back(token);
  }
  return array;
}

template <typename T>
inline std::vector<T> TextToArray(std::string const& text, std::uint64_t num_entry) {
  if (text.empty() && num_entry > 0) {
    TREELITE_LOG(FATAL) << "Cannot convert empty text into array";
  }
  std::vector<T> array;
  std::istringstream ss(text);
  std::string token;
  for (std::uint64_t i = 0; i < num_entry; ++i) {
    std::getline(ss, token, ' ');
    array.push_back(TextToNumber<T>(token));
  }
  return array;
}

enum Masks : std::uint8_t { kCategoricalMask = 1, kDefaultLeftMask = 2 };

enum class MissingType : std::uint8_t { kNone, kZero, kNaN };

struct LGBTree {
  int num_leaves;
  int num_cat;  // number of categorical splits
  std::vector<double> leaf_value;
  std::vector<std::int8_t> decision_type;
  std::vector<std::uint64_t> cat_boundaries;
  std::vector<std::uint32_t> cat_threshold;
  std::vector<int> split_feature;
  std::vector<double> threshold;
  std::vector<int> left_child;
  std::vector<int> right_child;
  std::vector<float> split_gain;
  std::vector<int> internal_count;
  std::vector<int> leaf_count;
};

inline bool GetDecisionType(std::int8_t decision_type, std::int8_t mask) {
  return (decision_type & mask) > 0;
}

inline MissingType GetMissingType(std::int8_t decision_type) {
  return static_cast<MissingType>((decision_type >> 2) & 3);
}

inline std::vector<std::uint32_t> BitsetToList(std::uint32_t const* bits, std::size_t nslots) {
  std::vector<std::uint32_t> result;
  std::size_t const nbits = nslots * 32;
  for (std::size_t i = 0; i < nbits; ++i) {
    std::size_t const i1 = i / 32;
    std::uint32_t const i2 = static_cast<std::uint32_t>(i % 32);
    if ((bits[i1] >> i2) & 1) {
      result.push_back(static_cast<std::uint32_t>(i));
    }
  }
  return result;
}

inline std::vector<std::string> LoadText(std::istream& fi) {
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(fi, line)) {
    treelite::model_loader::detail::StringTrimFromEnd(line);
    lines.push_back(line);
  }

  return lines;
}

inline std::unique_ptr<treelite::Model> ParseStream(std::istream& fi) {
  std::vector<LGBTree> lgb_trees_;
  int max_feature_idx_;
  int num_class_;
  bool average_output_;
  std::string obj_name_;
  std::vector<std::string> obj_param_;

  /* 1. Parse input stream */
  std::vector<std::string> lines = LoadText(fi);
  std::unordered_map<std::string, std::string> global_dict;
  std::vector<std::unordered_map<std::string, std::string>> tree_dict;

  bool in_tree = false;  // is current entry part of a tree?
  for (auto const& line : lines) {
    std::istringstream ss(line);
    std::string key, value, rest;
    std::getline(ss, key, '=');
    std::getline(ss, value, '=');
    std::getline(ss, rest);
    if (!rest.empty()) {
      value += "=";
      value += rest;
    }
    if (key == "Tree") {
      in_tree = true;
      tree_dict.emplace_back();
    } else {
      if (in_tree) {
        tree_dict.back()[key] = value;
      } else {
        global_dict[key] = value;
      }
    }
  }

  {
    auto it = global_dict.find("objective");
    if (it == global_dict.end()) {  // custom objective (fobj)
      obj_name_ = "custom";
    } else {
      auto obj_strs = Split(it->second, ' ');
      obj_name_ = obj_strs[0];
      obj_param_ = std::vector<std::string>(obj_strs.begin() + 1, obj_strs.end());
    }
    obj_name_ = treelite::model_loader::detail::lightgbm::CanonicalObjective(obj_name_);

    it = global_dict.find("max_feature_idx");
    TREELITE_CHECK(it != global_dict.end())
        << "Ill-formed LightGBM model file: need max_feature_idx";
    max_feature_idx_ = TextToNumber<int>(it->second);
    it = global_dict.find("num_class");
    TREELITE_CHECK(it != global_dict.end()) << "Ill-formed LightGBM model file: need num_class";
    num_class_ = TextToNumber<int>(it->second);

    it = global_dict.find("average_output");
    average_output_ = (it != global_dict.end());
  }

  for (auto const& dict : tree_dict) {
    lgb_trees_.emplace_back();
    LGBTree& tree = lgb_trees_.back();

    auto it = dict.find("num_leaves");
    TREELITE_CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need num_leaves";
    tree.num_leaves = TextToNumber<int>(it->second);

    it = dict.find("num_cat");
    TREELITE_CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need num_cat";
    tree.num_cat = TextToNumber<int>(it->second);

    it = dict.find("leaf_value");
    TREELITE_CHECK(it != dict.end() && !it->second.empty())
        << "Ill-formed LightGBM model file: need leaf_value";
    tree.leaf_value = TextToArray<double>(it->second, tree.num_leaves);

    it = dict.find("decision_type");
    if (tree.num_leaves <= 1) {
      tree.decision_type = std::vector<std::int8_t>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      if (it == dict.end()) {
        tree.decision_type = std::vector<std::int8_t>(tree.num_leaves - 1, 0);
      } else {
        TREELITE_CHECK(!it->second.empty())
            << "Ill-formed LightGBM model file: decision_type cannot be empty string";
        tree.decision_type = TextToArray<std::int8_t>(it->second, tree.num_leaves - 1);
      }
    }

    if (tree.num_cat > 0) {
      it = dict.find("cat_boundaries");
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
          << "Ill-formed LightGBM model file: need cat_boundaries";
      tree.cat_boundaries = TextToArray<std::uint64_t>(it->second, tree.num_cat + 1);
      it = dict.find("cat_threshold");
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
          << "Ill-formed LightGBM model file: need cat_threshold";
      tree.cat_threshold = TextToArray<std::uint32_t>(it->second, tree.cat_boundaries.back());
    }

    it = dict.find("split_feature");
    if (tree.num_leaves <= 1) {
      tree.split_feature = std::vector<int>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
          << "Ill-formed LightGBM model file: need split_feature";
      tree.split_feature = TextToArray<int>(it->second, tree.num_leaves - 1);
    }

    it = dict.find("threshold");
    if (tree.num_leaves <= 1) {
      tree.threshold = std::vector<double>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
          << "Ill-formed LightGBM model file: need threshold";
      tree.threshold = TextToArray<double>(it->second, tree.num_leaves - 1);
    }

    it = dict.find("split_gain");
    if (tree.num_leaves <= 1) {
      tree.split_gain = std::vector<float>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      if (it != dict.end()) {
        TREELITE_CHECK(!it->second.empty())
            << "Ill-formed LightGBM model file: split_gain cannot be empty string";
        tree.split_gain = TextToArray<float>(it->second, tree.num_leaves - 1);
      } else {
        tree.split_gain = std::vector<float>();
      }
    }

    it = dict.find("internal_count");
    if (tree.num_leaves <= 1) {
      tree.internal_count = std::vector<int>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      if (it != dict.end()) {
        TREELITE_CHECK(!it->second.empty())
            << "Ill-formed LightGBM model file: internal_count cannot be empty string";
        tree.internal_count = TextToArray<int>(it->second, tree.num_leaves - 1);
      } else {
        tree.internal_count = std::vector<int>();
      }
    }

    it = dict.find("leaf_count");
    if (tree.num_leaves == 0) {
      tree.leaf_count = std::vector<int>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 0);
      if (it != dict.end() && !it->second.empty()) {
        tree.leaf_count = TextToArray<int>(it->second, tree.num_leaves);
      } else {
        tree.leaf_count = std::vector<int>();
      }
    }

    it = dict.find("left_child");
    if (tree.num_leaves <= 1) {
      tree.left_child = std::vector<int>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
          << "Ill-formed LightGBM model file: need left_child";
      tree.left_child = TextToArray<int>(it->second, tree.num_leaves - 1);
    }

    it = dict.find("right_child");
    if (tree.num_leaves <= 1) {
      tree.right_child = std::vector<int>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
          << "Ill-formed LightGBM model file: need right_child";
      tree.right_child = TextToArray<int>(it->second, tree.num_leaves - 1);
    }
  }

  /* 2. Set model metadata */
  treelite::TaskType task_type;
  TREELITE_CHECK_LE(lgb_trees_.size(), std::numeric_limits<std::int32_t>::max())
      << "Too many trees";
  auto const num_tree = static_cast<std::int32_t>(lgb_trees_.size());
  std::vector<std::int32_t> class_id(num_tree, 0);
  if (num_class_ > 1) {
    // Multi-class classifier
    task_type = treelite::TaskType::kMultiClf;
    TREELITE_CHECK(obj_name_ == "multiclass" || obj_name_ == "multiclassova")
        << "Objective must be 'multiclass' or 'multiclassova' when num_class > 1";
    for (std::int32_t i = 0; i < num_tree; ++i) {
      class_id[i] = i % num_class_;
    }
  } else if (obj_name_ == "binary" || obj_name_ == "cross_entropy"
             || obj_name_ == "cross_entropy_lambda") {
    // Binary classifier
    task_type = treelite::TaskType::kBinaryClf;
  } else if (obj_name_ == "lambdarank" || obj_name_ == "rank_xendcg") {
    // Learning-to-rank
    task_type = treelite::TaskType::kLearningToRank;
  } else {
    // Regressor
    task_type = treelite::TaskType::kRegressor;
  }

  // Set correct prediction transform function, depending on objective function
  using treelite::model_builder::PostProcessorFunc;
  std::optional<PostProcessorFunc> postprocessor = std::nullopt;
  if (obj_name_ == "multiclass") {
    // Validate num_class parameter
    int num_class = -1;
    int tmp;
    for (auto const& str : obj_param_) {
      auto tokens = Split(str, ':');
      if (tokens.size() == 2 && tokens[0] == "num_class"
          && (tmp = TextToNumber<int>(tokens[1])) >= 0) {
        num_class = tmp;
        break;
      }
    }
    TREELITE_CHECK(num_class >= 0 && num_class == num_class_)
        << "Ill-formed LightGBM model file: not a valid multiclass objective";
    postprocessor = PostProcessorFunc{"softmax"};
  } else if (obj_name_ == "multiclassova") {
    // Validate num_class and alpha parameters
    int num_class = -1;
    float alpha = -1.0f;
    int tmp;
    float tmp2;
    for (auto const& str : obj_param_) {
      auto tokens = Split(str, ':');
      if (tokens.size() == 2) {
        if (tokens[0] == "num_class" && (tmp = TextToNumber<int>(tokens[1])) >= 0) {
          num_class = tmp;
        } else if (tokens[0] == "sigmoid" && (tmp2 = TextToNumber<float>(tokens[1])) > 0.0f) {
          alpha = tmp2;
        }
      }
    }
    TREELITE_CHECK(num_class >= 0 && num_class == num_class_ && alpha > 0.0f)
        << "Ill-formed LightGBM model file: not a valid multiclassova objective";
    postprocessor = PostProcessorFunc{"multiclass_ova", {{"sigmoid_alpha", alpha}}};
  } else if (obj_name_ == "binary") {
    // Validate alpha parameter
    float alpha = -1.0f;
    float tmp;
    for (auto const& str : obj_param_) {
      auto tokens = Split(str, ':');
      if (tokens.size() == 2 && tokens[0] == "sigmoid"
          && (tmp = TextToNumber<float>(tokens[1])) > 0.0f) {
        alpha = tmp;
        break;
      }
    }
    TREELITE_CHECK_GT(alpha, 0.0f)
        << "Ill-formed LightGBM model file: not a valid binary objective";
    postprocessor = PostProcessorFunc{"sigmoid", {{"sigmoid_alpha", alpha}}};
  } else if (obj_name_ == "cross_entropy") {
    postprocessor = PostProcessorFunc{"sigmoid", {{"sigmoid_alpha", 1.0}}};
  } else if (obj_name_ == "cross_entropy_lambda") {
    postprocessor = PostProcessorFunc{"logarithm_one_plus_exp"};
  } else if (obj_name_ == "poisson" || obj_name_ == "gamma" || obj_name_ == "tweedie") {
    postprocessor = PostProcessorFunc{"exponential"};
  } else if (obj_name_ == "regression" || obj_name_ == "regression_l1" || obj_name_ == "huber"
             || obj_name_ == "fair" || obj_name_ == "quantile" || obj_name_ == "mape") {
    // Regression family
    bool sqrt = (std::find(obj_param_.cbegin(), obj_param_.cend(), "sqrt") != obj_param_.cend());
    if (sqrt) {
      postprocessor = PostProcessorFunc{"signed_square"};
    } else {
      postprocessor = PostProcessorFunc{"identity"};
    }
  } else if (obj_name_ == "lambdarank" || obj_name_ == "rank_xendcg" || obj_name_ == "custom") {
    // Ranking family, or a custom user-defined objective
    postprocessor = PostProcessorFunc{"identity"};
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized objective: " << obj_name_;
  }
  TREELITE_CHECK(postprocessor.has_value());

  treelite::model_builder::Metadata metadata{static_cast<std::int32_t>(max_feature_idx_ + 1),
      task_type, average_output_, 1, {static_cast<std::int32_t>(num_class_)}, {1, 1}};
  treelite::model_builder::TreeAnnotation tree_annotation{
      num_tree, std::vector<std::int32_t>(num_tree, 0), class_id};
  auto builder = treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat64,
      treelite::TypeInfo::kFloat64, metadata, tree_annotation, postprocessor.value(),
      std::vector<double>(num_class_, 0.0));

  // Traverse trees
  for (auto const& lgb_tree : lgb_trees_) {
    builder->StartTree();

    // Assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    // We re-arrange nodes here, since LightGBM uses negative indices to distinguish leaf nodes
    // from internal nodes.
    std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
    if (lgb_tree.num_leaves == 0) {
      continue;
    } else if (lgb_tree.num_leaves == 1) {
      // A constant-value tree with a single root node that's also a leaf
      Q.emplace(-1, 0);
    } else {
      Q.emplace(0, 0);
    }
    while (!Q.empty()) {
      auto [old_node_id, new_node_id] = Q.front();
      Q.pop();
      builder->StartNode(new_node_id);
      if (old_node_id < 0) {  // leaf
        builder->LeafScalar(lgb_tree.leaf_value[~old_node_id]);
        if (!lgb_tree.leaf_count.empty()) {
          int const data_count = lgb_tree.leaf_count[~old_node_id];
          TREELITE_CHECK_GE(data_count, 0);
          builder->DataCount(data_count);
        }
      } else {  // non-leaf
        auto const split_index = static_cast<std::int32_t>(lgb_tree.split_feature[old_node_id]);
        auto const missing_type = GetMissingType(lgb_tree.decision_type[old_node_id]);
        int const left_child_old_id = lgb_tree.left_child[old_node_id];
        int const left_child_new_id = new_node_id * 2 + 1;
        int const right_child_old_id = lgb_tree.right_child[old_node_id];
        int const right_child_new_id = new_node_id * 2 + 2;

        if (GetDecisionType(lgb_tree.decision_type[old_node_id], kCategoricalMask)) {
          // Categorical split
          int const cat_idx = static_cast<int>(lgb_tree.threshold[old_node_id]);
          std::vector<std::uint32_t> const left_categories
              = BitsetToList(lgb_tree.cat_threshold.data() + lgb_tree.cat_boundaries[cat_idx],
                  lgb_tree.cat_boundaries[cat_idx + 1] - lgb_tree.cat_boundaries[cat_idx]);
          // For categorical splits, we ignore the missing type field. NaNs always get mapped to
          // the right child node.
          bool default_left = false;
          builder->CategoricalTest(split_index, default_left, left_categories, false,
              left_child_new_id, right_child_new_id);
        } else {
          // Numerical split
          auto const threshold = static_cast<double>(lgb_tree.threshold[old_node_id]);
          bool default_left
              = GetDecisionType(lgb_tree.decision_type[old_node_id], kDefaultLeftMask);
          bool const missing_value_to_zero = (missing_type != MissingType::kNaN);
          if (missing_value_to_zero) {
            // If missing_value_to_zero flag is true, all missing values get mapped to 0.0, so
            // we need to override the default_left flag
            default_left = 0.0 <= threshold;
          }
          builder->NumericalTest(split_index, threshold, default_left, treelite::Operator::kLE,
              left_child_new_id, right_child_new_id);
        }
        if (!lgb_tree.internal_count.empty()) {
          int const data_count = lgb_tree.internal_count[old_node_id];
          TREELITE_CHECK_GE(data_count, 0);
          builder->DataCount(data_count);
        }
        if (!lgb_tree.split_gain.empty()) {
          builder->Gain(lgb_tree.split_gain[old_node_id]);
        }
        Q.emplace(left_child_old_id, left_child_new_id);
        Q.emplace(right_child_old_id, right_child_new_id);
      }
      builder->EndNode();
    }
    builder->EndTree();
  }
  return builder->CommitModel();
}

}  // anonymous namespace
