/*!
 * Copyright 2017 by Contributors
 * \file lightgbm.cc
 * \brief Frontend for lightgbm model
 * \author Philip Cho
 */

#include <unordered_map>
#include <queue>
#include <dmlc/data.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

namespace {

treelite::Model ParseStream(dmlc::Stream* fi);

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(lightgbm);

Model LoadLightGBMModel(const char* filename) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename, "r"));
  return ParseStream(fi.get());
}

}  // namespace frontend
}  // namespace treelite

/* auxiliary data structures to interpret lightgbm model file */
namespace {

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

inline std::vector<std::string> LoadText(dmlc::Stream* fi) {
  const size_t bufsize = 16 * 1024 * 1024;  // 16 MB
  std::vector<char> buf(bufsize);

  std::vector<std::string> lines;

  size_t byte_read;

  std::string leftover = "";  // carry over between buffers
  while ((byte_read = fi->Read(&buf[0], sizeof(char) * bufsize)) > 0) {
    size_t i = 0;
    size_t tok_begin = 0;
    while (i < byte_read) {
      if (buf[i] == '\n' || buf[i] == '\r') {  // delimiter for lines
        if (tok_begin == 0 && leftover.length() + i > 0) {
          // first line in buffer
          lines.push_back(leftover + std::string(&buf[0], i));
          leftover = "";
        } else {
          lines.emplace_back(&buf[tok_begin], i - tok_begin);
        }
        // skip all delimiters afterwards
        for (; (buf[i] == '\n' || buf[i] == '\r') && i < byte_read; ++i) {}
        tok_begin = i;
      } else {
        ++i;
      }
    }
    // left-over string
    leftover += std::string(&buf[tok_begin], byte_read - tok_begin);
  }

  if (!leftover.empty()) {
    LOG(INFO)
      << "Warning: input file was not terminated with end-of-line character.";
    lines.push_back(leftover);
  }

  return lines;
}

inline treelite::Model ParseStream(dmlc::Stream* fi) {
  std::vector<LGBTree> lgb_trees_;
  int max_feature_idx_;
  int num_tree_per_iteration_;
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
    CHECK(rest.empty()) << "Ill-formed LightGBM model file";
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
    CHECK(it != global_dict.end())
      << "Ill-formed LightGBM model file: need objective";
    auto obj_strs = treelite::common::Split(it->second, ' ');
    obj_name_ = obj_strs[0];
    obj_param_ = std::vector<std::string>(obj_strs.begin() + 1, obj_strs.end());

    it = global_dict.find("max_feature_idx");
    CHECK(it != global_dict.end())
      << "Ill-formed LightGBM model file: need max_feature_idx";
    max_feature_idx_ = treelite::common::TextToNumber<int>(it->second);
    it = global_dict.find("num_tree_per_iteration");
    CHECK(it != global_dict.end())
      << "Ill-formed LightGBM model file: need num_tree_per_iteration";
    num_tree_per_iteration_ = treelite::common::TextToNumber<int>(it->second);

    it = global_dict.find("average_output");
    average_output_ = (it != global_dict.end());
  }

  for (const auto& dict : tree_dict) {
    lgb_trees_.emplace_back();
    LGBTree& tree = lgb_trees_.back();

    auto it = dict.find("num_leaves");
    CHECK(it != dict.end())
      << "Ill-formed LightGBM model file: need num_leaves";
    tree.num_leaves = treelite::common::TextToNumber<int>(it->second);

    it = dict.find("num_cat");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need num_cat";
    tree.num_cat = treelite::common::TextToNumber<int>(it->second);

    it = dict.find("leaf_value");
    CHECK(it != dict.end() && !it->second.empty())
      << "Ill-formed LightGBM model file: need leaf_value";
    tree.leaf_value
      = treelite::common::TextToArray<double>(it->second, tree.num_leaves);

    it = dict.find("decision_type");
    if (it == dict.end()) {
      tree.decision_type = std::vector<int8_t>(tree.num_leaves - 1, 0);
    } else {
      CHECK(tree.num_leaves - 1 == 0 || !it->second.empty())
        << "Ill-formed LightGBM model file: decision_type cannot be empty string";
      tree.decision_type
        = treelite::common::TextToArray<int8_t>(it->second,
                                                tree.num_leaves - 1);
    }

    if (tree.num_cat > 0) {
      it = dict.find("cat_boundaries");
      CHECK(it != dict.end() && !it->second.empty())
        << "Ill-formed LightGBM model file: need cat_boundaries";
      tree.cat_boundaries
        = treelite::common::TextToArray<uint64_t>(it->second, tree.num_cat + 1);
      it = dict.find("cat_threshold");
      CHECK(it != dict.end() && !it->second.empty())
        << "Ill-formed LightGBM model file: need cat_threshold";
      tree.cat_threshold
        = treelite::common::TextToArray<uint32_t>(it->second,
                                                  tree.cat_boundaries.back());
    }

    it = dict.find("split_feature");
    CHECK(it != dict.end() && (tree.num_leaves - 1 == 0 || !it->second.empty()))
      << "Ill-formed LightGBM model file: need split_feature";
    tree.split_feature
      = treelite::common::TextToArray<int>(it->second, tree.num_leaves - 1);

    it = dict.find("threshold");
    CHECK(it != dict.end() && (tree.num_leaves - 1 == 0 || !it->second.empty()))
      << "Ill-formed LightGBM model file: need threshold";
    tree.threshold
      = treelite::common::TextToArray<double>(it->second, tree.num_leaves - 1);

    it = dict.find("split_gain");
    if (it != dict.end()) {
      CHECK(tree.num_leaves - 1 == 0 || !it->second.empty())
        << "Ill-formed LightGBM model file: split_gain cannot be empty string";
      tree.split_gain
        = treelite::common::TextToArray<float>(it->second, tree.num_leaves - 1);
    } else {
      tree.split_gain.resize(tree.num_leaves - 1);
    }

    it = dict.find("internal_count");
    if (it != dict.end()) {
      CHECK(tree.num_leaves - 1 == 0 || !it->second.empty())
        << "Ill-formed LightGBM model file: internal_count cannot be empty string";
      tree.internal_count
        = treelite::common::TextToArray<int>(it->second, tree.num_leaves - 1);
    } else {
      tree.internal_count.resize(tree.num_leaves - 1);
    }

    it = dict.find("leaf_count");
    if (it != dict.end()) {
      CHECK(!it->second.empty())
        << "Ill-formed LightGBM model file: leaf_count cannot be empty string";
      tree.leaf_count
        = treelite::common::TextToArray<int>(it->second, tree.num_leaves);
    } else {
      tree.leaf_count.resize(tree.num_leaves);
    }

    it = dict.find("left_child");
    CHECK(it != dict.end() && (tree.num_leaves - 1 == 0 || !it->second.empty()))
      << "Ill-formed LightGBM model file: need left_child";
    tree.left_child
      = treelite::common::TextToArray<int>(it->second, tree.num_leaves - 1);

    it = dict.find("right_child");
    CHECK(it != dict.end() && (tree.num_leaves - 1 == 0 || !it->second.empty()))
      << "Ill-formed LightGBM model file: need right_child";
    tree.right_child
      = treelite::common::TextToArray<int>(it->second, tree.num_leaves - 1);
  }

  /* 2. Export model */
  treelite::Model model;
  model.num_feature = max_feature_idx_ + 1;
  model.num_output_group = num_tree_per_iteration_;
  if (model.num_output_group > 1) {
    // multiclass classification with gradient boosted trees
    CHECK(!average_output_)
      << "Ill-formed LightGBM model file: cannot use random forest mode "
      << "for multi-class classification";
    model.random_forest_flag = false;
  } else {
    model.random_forest_flag = average_output_;
  }

  // set correct prediction transform function, depending on objective function
  if (obj_name_ == "multiclass") {
    // validate num_class parameter
    int num_class = -1;
    int tmp;
    for (const auto& str : obj_param_) {
      auto tokens = treelite::common::Split(str, ':');
      if (tokens.size() == 2 && tokens[0] == "num_class"
        && (tmp = treelite::common::TextToNumber<int>(tokens[1])) >= 0) {
        num_class = tmp;
        break;
      }
    }
    CHECK(num_class >= 0 && num_class == model.num_output_group)
      << "Ill-formed LightGBM model file: not a valid multiclass objective";

    model.param.pred_transform = "softmax";
  } else if (obj_name_ == "multiclassova") {
    // validate num_class and alpha parameters
    int num_class = -1;
    float alpha = -1.0f;
    int tmp;
    float tmp2;
    for (const auto& str : obj_param_) {
      auto tokens = treelite::common::Split(str, ':');
      if (tokens.size() == 2) {
        if (tokens[0] == "num_class"
          && (tmp = treelite::common::TextToNumber<int>(tokens[1])) >= 0) {
          num_class = tmp;
        } else if (tokens[0] == "sigmoid"
         && (tmp2 = treelite::common::TextToNumber<float>(tokens[1])) > 0.0f) {
          alpha = tmp2;
        }
      }
    }
    CHECK(num_class >= 0 && num_class == model.num_output_group
          && alpha > 0.0f)
      << "Ill-formed LightGBM model file: not a valid multiclassova objective";

    model.param.pred_transform = "multiclass_ova";
    model.param.sigmoid_alpha = alpha;
  } else if (obj_name_ == "binary") {
    // validate alpha parameter
    float alpha = -1.0f;
    float tmp;
    for (const auto& str : obj_param_) {
      auto tokens = treelite::common::Split(str, ':');
      if (tokens.size() == 2 && tokens[0] == "sigmoid"
        && (tmp = treelite::common::TextToNumber<float>(tokens[1])) > 0.0f) {
        alpha = tmp;
        break;
      }
    }
    CHECK_GT(alpha, 0.0f)
      << "Ill-formed LightGBM model file: not a valid binary objective";

    model.param.pred_transform = "sigmoid";
    model.param.sigmoid_alpha = alpha;
  } else if (obj_name_ == "xentropy" || obj_name_ == "cross_entropy") {
    model.param.pred_transform = "sigmoid";
    model.param.sigmoid_alpha = 1.0f;
  } else if (obj_name_ == "xentlambda" || obj_name_ == "cross_entropy_lambda") {
    model.param.pred_transform = "logarithm_one_plus_exp";
  } else {
    model.param.pred_transform = "identity";
  }

  // traverse trees
  for (const auto& lgb_tree : lgb_trees_) {
    model.trees.emplace_back();
    treelite::Tree& tree = model.trees.back();
    tree.Init();

    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
    Q.push({0, 0});
    while (!Q.empty()) {
      int old_id, new_id;
      std::tie(old_id, new_id) = Q.front(); Q.pop();
      if (old_id < 0) {  // leaf
        const double leaf_value = lgb_tree.leaf_value[~old_id];
        const int data_count = lgb_tree.leaf_count[~old_id];
        tree[new_id].set_leaf(static_cast<treelite::tl_float>(leaf_value));
        CHECK_GE(data_count, 0);
        tree[new_id].set_data_count(static_cast<size_t>(data_count));
      } else {  // non-leaf
        const int data_count = lgb_tree.internal_count[old_id];
        const unsigned split_index =
          static_cast<unsigned>(lgb_tree.split_feature[old_id]);

        tree.AddChilds(new_id);
        if (GetDecisionType(lgb_tree.decision_type[old_id], kCategoricalMask)) {
          // categorical
          const int cat_idx = static_cast<int>(lgb_tree.threshold[old_id]);
          const std::vector<uint32_t> left_categories
            = BitsetToList(lgb_tree.cat_threshold.data()
                             + lgb_tree.cat_boundaries[cat_idx],
                           lgb_tree.cat_boundaries[cat_idx + 1]
                             - lgb_tree.cat_boundaries[cat_idx]);
          const auto missing_type
            = GetMissingType(lgb_tree.decision_type[old_id]);
          tree[new_id].set_categorical_split(split_index, false,
                                             (missing_type != MissingType::kNaN),
                                             left_categories);
        } else {
          // numerical
          const treelite::tl_float threshold =
            static_cast<treelite::tl_float>(lgb_tree.threshold[old_id]);
          const bool default_left
            = GetDecisionType(lgb_tree.decision_type[old_id], kDefaultLeftMask);
          const treelite::Operator cmp_op = treelite::Operator::kLE;
          tree[new_id].set_numerical_split(split_index, threshold,
                                           default_left, cmp_op);
        }
        CHECK_GE(data_count, 0);
        tree[new_id].set_data_count(static_cast<size_t>(data_count));
        tree[new_id].set_gain(static_cast<double>(lgb_tree.split_gain[old_id]));
        Q.push({lgb_tree.left_child[old_id], tree[new_id].cleft()});
        Q.push({lgb_tree.right_child[old_id], tree[new_id].cright()});
      }
    }
  }
  LOG(INFO) << "model.num_tree = " << model.trees.size();
  return model;
}

}  // anonymous namespace
