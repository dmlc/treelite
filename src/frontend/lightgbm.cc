/*!
 * Copyright 2017 by Contributors
 * \file lightgbm.cc
 * \brief Frontend for lightgbm model
 * \author Philip Cho
 */

#include <dmlc/data.h>
#include <treelite/tree.h>
#include <unordered_map>
#include <queue>
#include <cerrno>
#include <cstdlib>
#include <climits>

namespace {

treelite::Model ParseStream(dmlc::Stream* fi);

}  // namespace anonymous

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

enum class DecisionType : int8_t {
  numerical = 0, categorical = 1
};

struct LGBTree {
  int num_leaves;
  std::vector<double> leaf_value;
  std::vector<DecisionType> decision_type;
  std::vector<int> split_feature;
  std::vector<double> default_value;
  std::vector<double> threshold;
  std::vector<int> left_child;
  std::vector<int> right_child;
};

template <typename T>
inline T TextToEntry(const std::string& str) {
  static_assert(std::is_same<T, double>::value || std::is_same<T, int>::value,
                "unsupported data type for TextToEntry; use double or int");
}

template <>
inline double TextToEntry(const std::string& str) {
  errno = 0;
  char *endptr;
  double val = std::strtod(str.c_str(), &endptr);
  if (errno == ERANGE) {
    LOG(FATAL) << "Range error while converting string to double";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid floating-point number";
  }
  return val;
}

template <>
inline int TextToEntry(const std::string& str) {
  errno = 0;
  char *endptr;
  long int val = std::strtol(str.c_str(), &endptr, 10);
  if (errno == ERANGE || val < INT_MIN || val > INT_MAX) {
    LOG(FATAL) << "Range error while converting string to int";
  } else if (errno != 0) {
    LOG(FATAL) << "Unknown error";
  } else if (*endptr != '\0') {
    LOG(FATAL) << "String does not represent a valid integer";
  }
  return static_cast<int>(val);
}

template <typename T, typename S = T>
inline std::vector<S> TextToArray(const std::string& text, int num_entry) {
  std::vector<S> array;
  std::istringstream ss(text);
  std::string token;
  for (int i = 0; i < num_entry; ++i) {
    std::getline(ss, token, ' '); 
    array.push_back(static_cast<S>(TextToEntry<T>(token)));
  }
  return array;
}

inline std::vector<std::string> LoadText(dmlc::Stream* fi) {
  const size_t bufsize = 16 * 1024 * 1024;  // 16 MB
  std::vector<char> buf(bufsize);

  std::vector<std::string> lines;

  size_t byte_read;

  std::string leftover = "";  // carry over between buffers
  while ( (byte_read = fi->Read(&buf[0], sizeof(char) * bufsize)) > 0) {
    size_t i = 0;
    size_t tok_begin = 0;
    while (i < byte_read) {
      if (buf[i] == '\n' || buf[i] == '\r') {  // delimiter for lines
        if (tok_begin == 0 && leftover.length() + i > 0) {  // first line in buffer
          lines.push_back(leftover + std::string(&buf[0], i));
          leftover = "";
        } else {
          lines.emplace_back(&buf[tok_begin], i - tok_begin);
        }
        // skip all delimiters afterwards
        for (; (buf[i] == '\n' || buf[i] == '\r') && i < byte_read; ++i);
        tok_begin = i;
      } else {
        ++i;
      }
    }
    // left-over string
    leftover += std::string(&buf[tok_begin], byte_read - tok_begin);
  }

  if (!leftover.empty()) {
    LOG(INFO) << "Warning: input file was not terminated with end-of-line character.";
    lines.push_back(leftover);
  }

  return lines;
}

inline treelite::Model ParseStream(dmlc::Stream* fi) {
  std::vector<LGBTree> lgb_trees_;
  int max_feature_idx_;
  std::string obj_name_;

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
    CHECK(it != global_dict.end()) << "Ill-formed LightGBM model file: need objective";
    obj_name_ = it->second;
    it = global_dict.find("max_feature_idx");
    CHECK(it != global_dict.end()) << "Ill-formed LightGBM model file: need max_feature_idx";
    max_feature_idx_ = TextToEntry<int>(it->second);
  }

  for (const auto& dict : tree_dict) {
    lgb_trees_.emplace_back();
    LGBTree& tree = lgb_trees_.back();

    auto it = dict.find("num_leaves");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need num_leaves";
    tree.num_leaves = TextToEntry<int>(it->second);

    it = dict.find("leaf_value");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need leaf_value";
    tree.leaf_value = TextToArray<double>(it->second, tree.num_leaves);

    it = dict.find("decision_type");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need decision_type";
    if (it == dict.end()) {
      tree.decision_type = std::vector<DecisionType>(tree.num_leaves - 1, DecisionType::numerical);
    } else {
      tree.decision_type = TextToArray<int, DecisionType>(it->second, tree.num_leaves - 1);
    }

    it = dict.find("split_feature");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need split_feature";
    tree.split_feature = TextToArray<int>(it->second, tree.num_leaves - 1);

    it = dict.find("default_value");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need default_value";
    tree.default_value = TextToArray<double>(it->second, tree.num_leaves - 1);

    it = dict.find("threshold");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need threshold";
    tree.threshold = TextToArray<double>(it->second, tree.num_leaves - 1);

    it = dict.find("left_child");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need left_child";
    tree.left_child = TextToArray<int>(it->second, tree.num_leaves - 1);

    it = dict.find("right_child");
    CHECK(it != dict.end()) << "Ill-formed LightGBM model file: need right_child";
    tree.right_child = TextToArray<int>(it->second, tree.num_leaves - 1);
  }

  /* 2. Export model */
  treelite::Model model;
  model.num_features = max_feature_idx_ + 1;
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
        tree[new_id].set_leaf(static_cast<treelite::tl_float>(leaf_value));
      } else {  // non-leaf
        const unsigned split_index =
          static_cast<unsigned>(lgb_tree.split_feature[old_id]);
        const treelite::tl_float threshold = 
          static_cast<treelite::tl_float>(lgb_tree.threshold[old_id]);
        const treelite::tl_float default_value =
          static_cast<treelite::tl_float>(lgb_tree.default_value[old_id]);
        bool default_left;
        treelite::Operator cmp_op;
        switch (lgb_tree.decision_type[old_id]) {
          case DecisionType::numerical:
            cmp_op = treelite::Operator::kLE;
            default_left = (default_value <= threshold);
            break;
          case DecisionType::categorical:
            cmp_op = treelite::Operator::kEQ;
            default_left = (default_value == threshold);
            break;
          default:
            LOG(FATAL) << "invalid value for decision type";
            cmp_op = treelite::Operator::kEQ;  // only remove warning
            default_left = true;  // ditto
        }
        tree.AddChilds(new_id);
        tree[new_id].set_split(split_index, threshold, default_left, cmp_op);
        Q.push({lgb_tree.left_child[old_id], tree[new_id].cleft()});
        Q.push({lgb_tree.right_child[old_id], tree[new_id].cright()});
      }
    }
  }
  return model;
}

}  // namespace anonymous
