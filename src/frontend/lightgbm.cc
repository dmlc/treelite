/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file lightgbm.cc
 * \brief Frontend for LightGBM model
 * \author Hyunsu Cho
 */

#include <treelite/logging.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <unordered_map>
#include <limits>
#include <queue>
#include <string>
#include <fstream>

namespace {

inline std::unique_ptr<treelite::Model> ParseStream(std::istream& fi);

}  // anonymous namespace

namespace treelite {
namespace frontend {


std::unique_ptr<treelite::Model> LoadLightGBMModel(const char* filename) {
  std::ifstream fi(filename, std::ios::in);
  return ParseStream(fi);
}

}  // namespace frontend
}  // namespace treelite

/* auxiliary data structures to interpret lightgbm model file */
namespace {

template <typename T>
inline T TextToNumber(const std::string& str) {
  static_assert(std::is_same<T, float>::value
                || std::is_same<T, double>::value
                || std::is_same<T, int>::value
                || std::is_same<T, int8_t>::value
                || std::is_same<T, uint32_t>::value
                || std::is_same<T, uint64_t>::value,
                "unsupported data type for TextToNumber; use float, double, "
                "int, int8_t, uint32_t, or uint64_t");
}

template <>
inline float TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
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
inline double TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
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
inline int TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
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
inline int8_t TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  auto val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < std::numeric_limits<int8_t>::min()
      || val > std::numeric_limits<int8_t>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to int8_t";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int8_t>(val);
}

template <>
inline uint32_t TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  auto val = std::strtoul(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val > std::numeric_limits<uint32_t>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to uint32_t";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<uint32_t>(val);
}

template <>
inline uint64_t TextToNumber(const std::string& str) {
  errno = 0;
  char *endptr;
  auto val = std::strtoull(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val > std::numeric_limits<uint64_t>::max()) {
    TREELITE_LOG(FATAL) << "Range error while converting string to uint64_t";
  } else if (errno != 0) {
    TREELITE_LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    TREELITE_LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<uint64_t>(val);
}

inline std::vector<std::string> Split(const std::string& text, char delim) {
  std::vector<std::string> array;
  std::istringstream ss(text);
  std::string token;
  while (std::getline(ss, token, delim)) {
    array.push_back(token);
  }
  return array;
}

template <typename T>
inline std::vector<T> TextToArray(const std::string& text, int num_entry) {
  if (text.empty() && num_entry > 0) {
    TREELITE_LOG(FATAL) << "Cannot convert empty text into array";
  }
  std::vector<T> array;
  std::istringstream ss(text);
  std::string token;
  for (int i = 0; i < num_entry; ++i) {
    std::getline(ss, token, ' ');
    array.push_back(TextToNumber<T>(token));
  }
  return array;
}

enum Masks : uint8_t {
  kCategoricalMask = 1,
  kDefaultLeftMask = 2
};

enum class MissingType : uint8_t {
  kNone,
  kZero,
  kNaN
};

struct LGBTree {
  int num_leaves;
  int num_cat;  // number of categorical splits
  std::vector<double> leaf_value;
  std::vector<int8_t> decision_type;
  std::vector<uint64_t> cat_boundaries;
  std::vector<uint32_t> cat_threshold;
  std::vector<int> split_feature;
  std::vector<double> threshold;
  std::vector<int> left_child;
  std::vector<int> right_child;
  std::vector<float> split_gain;
  std::vector<int> internal_count;
  std::vector<int> leaf_count;
};

inline bool GetDecisionType(int8_t decision_type, int8_t mask) {
  return (decision_type & mask) > 0;
}

inline MissingType GetMissingType(int8_t decision_type) {
  return static_cast<MissingType>((decision_type >> 2) & 3);
}

inline std::vector<uint32_t> BitsetToList(const uint32_t* bits,
                                          size_t nslots) {
  std::vector<uint32_t> result;
  const size_t nbits = nslots * 32;
  for (size_t i = 0; i < nbits; ++i) {
    const size_t i1 = i / 32;
    const uint32_t i2 = static_cast<uint32_t>(i % 32);
    if ((bits[i1] >> i2) & 1) {
      result.push_back(static_cast<uint32_t>(i));
    }
  }
  return result;
}

inline std::vector<std::string> LoadText(std::istream& fi) {
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(fi, line)) {
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
  for (const auto& line : lines) {
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

    it = global_dict.find("max_feature_idx");
    TREELITE_CHECK(it != global_dict.end())
      << "Ill-formed LightGBM model file: need max_feature_idx";
    max_feature_idx_ = TextToNumber<int>(it->second);
    it = global_dict.find("num_class");
    TREELITE_CHECK(it != global_dict.end())
      << "Ill-formed LightGBM model file: need num_class";
    num_class_ = TextToNumber<int>(it->second);

    it = global_dict.find("average_output");
    average_output_ = (it != global_dict.end());
  }

  for (const auto& dict : tree_dict) {
    lgb_trees_.emplace_back();
    LGBTree& tree = lgb_trees_.back();

    auto it = dict.find("num_leaves");
    TREELITE_CHECK(it != dict.end())
      << "Ill-formed LightGBM model file: need num_leaves";
    tree.num_leaves = TextToNumber<int>(it->second);

    it = dict.find("num_cat");
    TREELITE_CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need num_cat";
    tree.num_cat = TextToNumber<int>(it->second);

    it = dict.find("leaf_value");
    TREELITE_CHECK(it != dict.end() && !it->second.empty())
      << "Ill-formed LightGBM model file: need leaf_value";
    tree.leaf_value
      = TextToArray<double>(it->second, tree.num_leaves);

    it = dict.find("decision_type");
    if (tree.num_leaves <= 1) {
      tree.decision_type = std::vector<int8_t>();
    } else {
      TREELITE_CHECK_GT(tree.num_leaves, 1);
      if (it == dict.end()) {
        tree.decision_type = std::vector<int8_t>(tree.num_leaves - 1, 0);
      } else {
        TREELITE_CHECK(!it->second.empty())
          << "Ill-formed LightGBM model file: decision_type cannot be empty string";
        tree.decision_type = TextToArray<int8_t>(it->second, tree.num_leaves - 1);
      }
    }

    if (tree.num_cat > 0) {
      it = dict.find("cat_boundaries");
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
        << "Ill-formed LightGBM model file: need cat_boundaries";
      tree.cat_boundaries
        = TextToArray<uint64_t>(it->second, tree.num_cat + 1);
      it = dict.find("cat_threshold");
      TREELITE_CHECK(it != dict.end() && !it->second.empty())
        << "Ill-formed LightGBM model file: need cat_threshold";
      tree.cat_threshold
        = TextToArray<uint32_t>(it->second, static_cast<uint32_t>(tree.cat_boundaries.back()));
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

  /* 2. Export model */
  std::unique_ptr<treelite::Model> model_ptr = treelite::Model::Create<double, double>();
  auto* model = dynamic_cast<treelite::ModelImpl<double, double>*>(model_ptr.get());
  model->num_feature = max_feature_idx_ + 1;
  model->average_tree_output = average_output_;
  if (num_class_ > 1) {
    // multi-class classifier
    model->task_type = treelite::TaskType::kMultiClfGrovePerClass;
    model->task_param.grove_per_class = true;
  } else {
    // binary classifier or regressor
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
  }
  model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
  model->task_param.num_class = num_class_;
  model->task_param.leaf_vector_size = 1;

  // set correct prediction transform function, depending on objective function
  if (obj_name_ == "multiclass") {
    // validate num_class parameter
    int num_class = -1;
    int tmp;
    for (const auto& str : obj_param_) {
      auto tokens = Split(str, ':');
      if (tokens.size() == 2 && tokens[0] == "num_class"
        && (tmp = TextToNumber<int>(tokens[1])) >= 0) {
        num_class = tmp;
        break;
      }
    }
    TREELITE_CHECK(num_class >= 0 && static_cast<size_t>(num_class) == model->task_param.num_class)
      << "Ill-formed LightGBM model file: not a valid multiclass objective";

    std::strncpy(model->param.pred_transform, "softmax", sizeof(model->param.pred_transform));
  } else if (obj_name_ == "multiclassova") {
    // validate num_class and alpha parameters
    int num_class = -1;
    float alpha = -1.0f;
    int tmp;
    float tmp2;
    for (const auto& str : obj_param_) {
      auto tokens = Split(str, ':');
      if (tokens.size() == 2) {
        if (tokens[0] == "num_class"
          && (tmp = TextToNumber<int>(tokens[1])) >= 0) {
          num_class = tmp;
        } else if (tokens[0] == "sigmoid"
         && (tmp2 = TextToNumber<float>(tokens[1])) > 0.0f) {
          alpha = tmp2;
        }
      }
    }
    TREELITE_CHECK(num_class >= 0 && static_cast<size_t>(num_class) == model->task_param.num_class
                   && alpha > 0.0f)
      << "Ill-formed LightGBM model file: not a valid multiclassova objective";

    std::strncpy(model->param.pred_transform, "multiclass_ova",
                 sizeof(model->param.pred_transform));
    model->param.sigmoid_alpha = alpha;
  } else if (obj_name_ == "binary") {
    // validate alpha parameter
    float alpha = -1.0f;
    float tmp;
    for (const auto& str : obj_param_) {
      auto tokens = Split(str, ':');
      if (tokens.size() == 2 && tokens[0] == "sigmoid"
        && (tmp = TextToNumber<float>(tokens[1])) > 0.0f) {
        alpha = tmp;
        break;
      }
    }
    TREELITE_CHECK_GT(alpha, 0.0f)
      << "Ill-formed LightGBM model file: not a valid binary objective";

    std::strncpy(model->param.pred_transform, "sigmoid", sizeof(model->param.pred_transform));
    model->param.sigmoid_alpha = alpha;
  } else if (obj_name_ == "xentropy" || obj_name_ == "cross_entropy") {
    std::strncpy(model->param.pred_transform, "sigmoid", sizeof(model->param.pred_transform));
    model->param.sigmoid_alpha = 1.0f;
  } else if (obj_name_ == "xentlambda" || obj_name_ == "cross_entropy_lambda") {
    std::strncpy(model->param.pred_transform, "logarithm_one_plus_exp",
                 sizeof(model->param.pred_transform));
  } else if (obj_name_ == "poisson" || obj_name_ == "gamma" || obj_name_ == "tweedie") {
    std::strncpy(model->param.pred_transform, "exponential",
                 sizeof(model->param.pred_transform));
  } else if (obj_name_ == "regression" || obj_name_ == "regression_l1" || obj_name_ == "huber"
             || obj_name_ == "fair" || obj_name_ == "quantile" || obj_name_ == "mape") {
    // Regression family
    bool sqrt = (std::find(obj_param_.cbegin(), obj_param_.cend(), "sqrt") != obj_param_.cend());
    if (sqrt) {
      std::strncpy(model->param.pred_transform, "signed_square",
                   sizeof(model->param.pred_transform));
    } else {
      std::strncpy(model->param.pred_transform, "identity",
                   sizeof(model->param.pred_transform));
    }
  } else if (obj_name_ == "lambdarank" || obj_name_ == "rank_xendcg" || obj_name_ == "custom") {
    // Ranking family, or a custom user-defined objective
    std::strncpy(model->param.pred_transform, "identity",
                 sizeof(model->param.pred_transform));
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized objective: " << obj_name_;
  }

  // traverse trees
  for (const auto& lgb_tree : lgb_trees_) {
    model->trees.emplace_back();
    treelite::Tree<double, double>& tree = model->trees.back();
    tree.Init();

    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
    if (lgb_tree.num_leaves == 0) {
      continue;
    } else if (lgb_tree.num_leaves == 1) {
      // A constant-value tree with a single root node that's also a leaf
      Q.push({-1, 0});
    } else {
      Q.push({0, 0});
    }
    while (!Q.empty()) {
      int old_id, new_id;
      std::tie(old_id, new_id) = Q.front(); Q.pop();
      if (old_id < 0) {  // leaf
        const double leaf_value = lgb_tree.leaf_value[~old_id];
        tree.SetLeaf(new_id, static_cast<double>(leaf_value));
        if (!lgb_tree.leaf_count.empty()) {
          const int data_count = lgb_tree.leaf_count[~old_id];
          TREELITE_CHECK_GE(data_count, 0);
          tree.SetDataCount(new_id, static_cast<size_t>(data_count));
        }
      } else {  // non-leaf
        const auto split_index = static_cast<unsigned>(lgb_tree.split_feature[old_id]);
        const auto missing_type = GetMissingType(lgb_tree.decision_type[old_id]);

        tree.AddChilds(new_id);
        if (GetDecisionType(lgb_tree.decision_type[old_id], kCategoricalMask)) {
          // categorical split
          const int cat_idx = static_cast<int>(lgb_tree.threshold[old_id]);
          const std::vector<uint32_t> left_categories
            = BitsetToList(lgb_tree.cat_threshold.data()
                             + lgb_tree.cat_boundaries[cat_idx],
                           lgb_tree.cat_boundaries[cat_idx + 1]
                             - lgb_tree.cat_boundaries[cat_idx]);
          // For categorical splits, we ignore the missing type field. NaNs always get mapped to
          // the right child node.
          bool default_left = false;
          tree.SetCategoricalSplit(new_id, split_index, default_left, left_categories, false);
        } else {
          // numerical split
          const auto threshold = static_cast<double>(lgb_tree.threshold[old_id]);
          bool default_left
            = GetDecisionType(lgb_tree.decision_type[old_id], kDefaultLeftMask);
          const treelite::Operator cmp_op = treelite::Operator::kLE;
          const bool missing_value_to_zero = (missing_type != MissingType::kNaN);
          if (missing_value_to_zero) {
            // If missing_value_to_zero flag is true, all missing values get mapped to 0.0, so
            // we need to override the default_left flag
            default_left = 0.0 <= threshold;
          }
          tree.SetNumericalSplit(new_id, split_index, threshold, default_left, cmp_op);
        }
        if (!lgb_tree.internal_count.empty()) {
          const int data_count = lgb_tree.internal_count[old_id];
          TREELITE_CHECK_GE(data_count, 0);
          tree.SetDataCount(new_id, static_cast<size_t>(data_count));
        }
        if (!lgb_tree.split_gain.empty()) {
          tree.SetGain(new_id, static_cast<double>(lgb_tree.split_gain[old_id]));
        }
        Q.push({lgb_tree.left_child[old_id], tree.LeftChild(new_id)});
        Q.push({lgb_tree.right_child[old_id], tree.RightChild(new_id)});
      }
    }
  }
  return model_ptr;
}

}  // anonymous namespace
