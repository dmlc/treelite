/*!
 * Copyright 2017 by Contributors
 * \file recursive.cc
 * \brief Recursive compiler
 * \author Philip Cho
 */

#include <treelite/common.h>
#include <treelite/annotator.h>
#include <treelite/compiler.h>
#include <treelite/tree.h>
#include <treelite/semantic.h>
#include <dmlc/registry.h>
#include <queue>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <cmath>
#include "param.h"
#include "pred_transform.h"

namespace {

class NumericSplitCondition : public treelite::semantic::Condition {
 public:
  using NumericAdapter
    = std::function<std::string(treelite::Operator, unsigned,
                                treelite::tl_float)>;
  explicit NumericSplitCondition(const treelite::Tree::Node& node,
                                 const NumericAdapter& numeric_adapter)
   : split_index(node.split_index()), default_left(node.default_left()),
     op(node.comparison_op()), threshold(node.threshold()),
     numeric_adapter(numeric_adapter) {}
  explicit NumericSplitCondition(const treelite::Tree::Node& node,
                                 NumericAdapter&& numeric_adapter)
   : split_index(node.split_index()), default_left(node.default_left()),
     op(node.comparison_op()), threshold(node.threshold()),
     numeric_adapter(std::move(numeric_adapter)) {}
  CLONEABLE_BOILERPLATE(NumericSplitCondition)
  inline std::string Compile() const override {
    const std::string bitmap
      = std::string("data[") + std::to_string(split_index) + "].missing != -1";
    return ((default_left) ?  (std::string("!(") + bitmap + ") || ")
                            : (std::string(" (") + bitmap + ") && "))
            + numeric_adapter(op, split_index, threshold);
  }

 private:
  unsigned split_index;
  bool default_left;
  treelite::Operator op;
  treelite::tl_float threshold;
  NumericAdapter numeric_adapter;
};

class CategoricalSplitCondition : public treelite::semantic::Condition {
 public:
  explicit CategoricalSplitCondition(const treelite::Tree::Node& node)
   : split_index(node.split_index()), default_left(node.default_left()),
     categorical_bitmap(to_bitmap(node.left_categories())) {}
  CLONEABLE_BOILERPLATE(CategoricalSplitCondition)
  inline std::string Compile() const override {
    const std::string bitmap
      = std::string("data[") + std::to_string(split_index) + "].missing != -1";
    CHECK_GE(categorical_bitmap.size(), 1);
    std::ostringstream comp;
    comp << "(tmp = (unsigned int)(data[" << split_index << "].fvalue) ), "
         << "(tmp >= 0 && tmp < 64 && (( (uint64_t)"
         << categorical_bitmap[0] << "U >> tmp) & 1) )";
    for (size_t i = 1; i < categorical_bitmap.size(); ++i) {
      comp << " || (tmp >= " << (i * 64)
           << " && tmp < " << ((i + 1) * 64)
           << " && (( (uint64_t)" << categorical_bitmap[i]
           << "U >> (tmp - " << (i * 64) << ") ) & 1) )";
    }
    bool all_zeros = true;
    for (uint64_t e : categorical_bitmap) {
      all_zeros &= (e == 0);
    }
    return ((default_left) ?  (std::string("!(") + bitmap + ") || (")
                            : (std::string(" (") + bitmap + ") && ("))
            + (all_zeros ? std::string("0") : comp.str()) + ")";
  }

 private:
  unsigned split_index;
  bool default_left;
  std::vector<uint64_t> categorical_bitmap;

  inline std::vector<uint64_t> to_bitmap(const std::vector<uint32_t>& left_categories) const {
    const size_t num_left_categories = left_categories.size();
    const uint32_t max_left_category = left_categories[num_left_categories - 1];
    std::vector<uint64_t> bitmap((max_left_category + 1 + 63) / 64, 0);
    for (size_t i = 0; i < left_categories.size(); ++i) {
      const uint32_t cat = left_categories[i];
      const size_t idx = cat / 64;
      const uint32_t offset = cat % 64;
      bitmap[idx] |= (static_cast<uint64_t>(1) << offset);
    }
    return bitmap;
  }
};

struct GroupPolicy {
  void Init(const treelite::Model& model);

  std::string GroupQueryFunction() const;
  std::string Accumulator() const;
  std::string AccumulateTranslationUnit(size_t unit_id) const;
  std::vector<std::string> AccumulateLeaf(const treelite::Tree::Node& node,
                                          size_t tree_id) const;
  std::vector<std::string> Return() const;
  std::vector<std::string> FinalReturn(size_t num_tree, float global_bias) const;
  std::string Prototype() const;
  std::string PrototypeTranslationUnit(size_t unit_id) const;

  int num_output_group;
  bool random_forest_flag;
};

}  // namespace anonymous

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(recursive);

std::vector<std::vector<tl_float>> ExtractCutPoints(const Model& model);

struct Metadata {
  int num_feature;
  std::vector<std::vector<tl_float>> cut_pts;
  std::vector<bool> is_categorical;

  inline void Init(const Model& model, bool extract_cut_pts = false) {
    num_feature = model.num_feature;
    is_categorical.clear();
    is_categorical.resize(num_feature, false);
    for (const Tree& tree : model.trees) {
      for (unsigned e : tree.GetCategoricalFeatures()) {
        is_categorical[e] = true;
      }
    }
    if (extract_cut_pts) {
      cut_pts = std::move(ExtractCutPoints(model));
    }
  }
};

template <typename QuantizePolicy>
class RecursiveCompiler : public Compiler, private QuantizePolicy {
 public:
  explicit RecursiveCompiler(const CompilerParam& param)
    : param(param) {
    if (param.verbose > 0) {
      LOG(INFO) << "Using RecursiveCompiler";
    }
  }

  using SemanticModel = semantic::SemanticModel;
  using TranslationUnit = semantic::TranslationUnit;
  using CodeBlock = semantic::CodeBlock;
  using PlainBlock = semantic::PlainBlock;
  using FunctionBlock = semantic::FunctionBlock;
  using SequenceBlock = semantic::SequenceBlock;
  using IfElseBlock = semantic::IfElseBlock;
  using Condition = semantic::Condition;

  SemanticModel Compile(const Model& model) override {
    Metadata info;
    info.Init(model, QuantizePolicy::QuantizeFlag());
    QuantizePolicy::Init(std::move(info));
    group_policy.Init(model);

    std::vector<std::vector<size_t>> annotation;
    bool annotate = false;
    if (param.annotate_in != "NULL") {
      BranchAnnotator annotator;
      std::unique_ptr<dmlc::Stream> fi(
        dmlc::Stream::Create(param.annotate_in.c_str(), "r"));
      annotator.Load(fi.get());
      annotation = annotator.Get();
      annotate = true;
      if (param.verbose > 0) {
        LOG(INFO) << "Using branch annotation file `"
                  << param.annotate_in << "'";
      }
    }

    SemanticModel semantic_model;
    SequenceBlock sequence;
    sequence.Reserve(model.trees.size() + 3);
    sequence.PushBack(PlainBlock(group_policy.Accumulator()));
    sequence.PushBack(PlainBlock(QuantizePolicy::Preprocessing()));
    if (param.parallel_comp > 0) {
      LOG(INFO) << "Parallel compilation enabled; member trees will be "
                << "divided into " << param.parallel_comp
                << " translation units.";
      const size_t nunit = param.parallel_comp;  // # translation units
      for (size_t unit_id = 0; unit_id < nunit; ++unit_id) {
        sequence.PushBack(PlainBlock(
                             group_policy.AccumulateTranslationUnit(unit_id)));
      }
    } else {
      LOG(INFO) << "Parallel compilation disabled; all member trees will be "
                << "dump to a single source file. This may increase "
                << "compilation time and memory usage.";
      for (size_t tree_id = 0; tree_id < model.trees.size(); ++tree_id) {
        const Tree& tree = model.trees[tree_id];
        if (!annotation.empty()) {
          sequence.PushBack(common::MoveUniquePtr(WalkTree(tree, tree_id,
                                                        annotation[tree_id])));
        } else {
          sequence.PushBack(common::MoveUniquePtr(WalkTree(tree, tree_id,
                                                           {})));
        }
      }
    }
    sequence.PushBack(PlainBlock(group_policy.FinalReturn(model.trees.size(),
                                                     model.param.global_bias)));

    FunctionBlock query_func("size_t get_num_output_group(void)",
                             PlainBlock(group_policy.GroupQueryFunction()),
                             &semantic_model.function_registry, true);
    FunctionBlock query_func2("size_t get_num_feature(void)",
                              PlainBlock(std::string("return ") +
                                         std::to_string(model.num_feature)+";"),
                              &semantic_model.function_registry, true);
    FunctionBlock pred_transform_func(PredTransformPrototype(false),
                             PlainBlock(PredTransformFunction(model, false)),
                             &semantic_model.function_registry, true);
    FunctionBlock pred_transform_batch_func(PredTransformPrototype(true),
                             PlainBlock(PredTransformFunction(model, true)),
                             &semantic_model.function_registry, true);
    FunctionBlock main_func(group_policy.Prototype(),
      std::move(sequence), &semantic_model.function_registry, true);
    SequenceBlock main_file;
    main_file.Reserve(5);
    main_file.PushBack(std::move(query_func));
    main_file.PushBack(std::move(query_func2));
    main_file.PushBack(std::move(pred_transform_func));
    main_file.PushBack(std::move(pred_transform_batch_func));
    main_file.PushBack(std::move(main_func));
    auto file_preamble = QuantizePolicy::ConstantsPreamble();
    semantic_model.units.emplace_back(PlainBlock(file_preamble),
                                      std::move(main_file));

    if (param.parallel_comp > 0) {
      const size_t nunit = param.parallel_comp;
      const size_t unit_size = (model.trees.size() + nunit - 1) / nunit;
      for (size_t unit_id = 0; unit_id < nunit; ++unit_id) {
        const size_t tree_begin = unit_id * unit_size;
        const size_t tree_end = std::min((unit_id + 1) * unit_size,
                                         model.trees.size());
        SequenceBlock unit_seq;
        if (tree_begin < tree_end) {
          unit_seq.Reserve(tree_end - tree_begin + 2);
        }
        unit_seq.PushBack(PlainBlock(group_policy.Accumulator()));
        for (size_t tree_id = tree_begin; tree_id < tree_end; ++tree_id) {
          const Tree& tree = model.trees[tree_id];
          if (!annotation.empty()) {
            unit_seq.PushBack(common::MoveUniquePtr(WalkTree(tree, tree_id,
                                                        annotation[tree_id])));
          } else {
            unit_seq.PushBack(common::MoveUniquePtr(WalkTree(tree, tree_id,
                                                            {})));
          }
        }
        unit_seq.PushBack(PlainBlock(group_policy.Return()));
        FunctionBlock unit_func(group_policy.PrototypeTranslationUnit(unit_id),
                                std::move(unit_seq),
                                &semantic_model.function_registry);
        semantic_model.units.emplace_back(PlainBlock(), std::move(unit_func));
      }
    }
    std::vector<std::string> header{"#include <stdlib.h>",
                                    "#include <string.h>",
                                    "#include <math.h>",
                                    "#include <stdint.h>"};
    common::TransformPushBack(&header, QuantizePolicy::CommonHeader(),
                              [] (std::string line) { return line; });
    if (annotate) {
      header.emplace_back();
#if defined(__clang__) || defined(__GNUC__)
      // only gcc and clang support __builtin_expect intrinsic
      header.emplace_back("#define LIKELY(x)     __builtin_expect(!!(x), 1)");
      header.emplace_back("#define UNLIKELY(x)   __builtin_expect(!!(x), 0)");
#else
      header.emplace_back("#define LIKELY(x) (x)");
      header.emplace_back("#define UNLIKELY(x) (x)");
#endif
    }
    semantic_model.common_header
             = std::move(common::make_unique<PlainBlock>(header));
    return semantic_model;
  }

 private:
  CompilerParam param;
  GroupPolicy group_policy;

  std::unique_ptr<CodeBlock> WalkTree(const Tree& tree, size_t tree_id,
                                      const std::vector<size_t>& counts) const {
    return WalkTree_(tree, tree_id, counts, 0);
  }

  std::unique_ptr<CodeBlock> WalkTree_(const Tree& tree, size_t tree_id,
                                       const std::vector<size_t>& counts,
                                       int nid) const {
    using semantic::BranchHint;
    const Tree::Node& node = tree[nid];
    if (node.is_leaf()) {
      return std::unique_ptr<CodeBlock>(new PlainBlock(
        group_policy.AccumulateLeaf(node, tree_id)));
    } else {
      BranchHint branch_hint = BranchHint::kNone;
      if (!counts.empty()) {
        const size_t left_count = counts[node.cleft()];
        const size_t right_count = counts[node.cright()];
        branch_hint = (left_count > right_count) ? BranchHint::kLikely
                                                 : BranchHint::kUnlikely;
      }
      std::unique_ptr<Condition> condition(nullptr);
      if (node.split_type() == SplitFeatureType::kNumerical) {
        condition = common::make_unique<NumericSplitCondition>(node,
                                        QuantizePolicy::NumericAdapter());
      } else {
        condition = common::make_unique<CategoricalSplitCondition>(node);
      }
      return std::unique_ptr<CodeBlock>(new IfElseBlock(
        common::MoveUniquePtr(condition),
        common::MoveUniquePtr(WalkTree_(tree, tree_id, counts, node.cleft())),
        common::MoveUniquePtr(WalkTree_(tree, tree_id, counts, node.cright())),
        branch_hint)
      );
    }
  }
};

class MetadataStore {
 protected:
  void Init(const Metadata& info) {
    this->info = info;
  }
  void Init(Metadata&& info) {
    this->info = std::move(info);
  }
  const Metadata& GetInfo() const {
    return info;
  }
  MetadataStore() = default;
  MetadataStore(const MetadataStore& other) = default;
  MetadataStore(MetadataStore&& other) = default;
 private:
  Metadata info;
};

class NoQuantize : private MetadataStore {
 protected:
  template <typename... Args>
  void Init(Args&&... args) {
    MetadataStore::Init(std::forward<Args>(args)...);
  }
  NumericSplitCondition::NumericAdapter NumericAdapter() const {
    return [] (Operator op, unsigned split_index, tl_float threshold) {
      std::ostringstream oss;
      if (!std::isfinite(threshold)) {
        // According to IEEE 754, the result of comparison [lhs] < infinity
        // must be identical for all finite [lhs]. Same goes for operator >.
        oss << (semantic::CompareWithOp(0.0, op, threshold) ? "1" : "0");
      } else {
        // to restore default precision
        const std::streamsize ss = std::cout.precision();
        oss << "data[" << split_index << "].fvalue "
            << semantic::OpName(op) << " "
            << std::setprecision(std::numeric_limits<tl_float>::digits10 + 2)
            << threshold
            << std::setprecision(ss);
      }
      return oss.str();
    };
  }
  std::vector<std::string> CommonHeader() const {
    return {"",
            "union Entry {",
            "  int missing;",
            "  float fvalue;",
            "};"};
  }
  std::vector<std::string> ConstantsPreamble() const {
    return {};
  }
  std::vector<std::string> Preprocessing() const {
    return {};
  }
  bool QuantizeFlag() const {
    return false;
  }
};

class Quantize : private MetadataStore {
 protected:
  template <typename... Args>
  void Init(Args&&... args) {
    MetadataStore::Init(std::forward<Args>(args)...);
    quant_preamble = {
      std::string("for (int i = 0; i < ")
      + std::to_string(GetInfo().num_feature) + "; ++i) {",
      "  if (data[i].missing != -1 && !is_categorical[i]) {",
      "    data[i].qvalue = quantize(data[i].fvalue, i);",
      "  }",
      "}"};
  }
  NumericSplitCondition::NumericAdapter NumericAdapter() const {
    const std::vector<std::vector<tl_float>>& cut_pts = GetInfo().cut_pts;
    return [&cut_pts] (Operator op, unsigned split_index,
                       tl_float threshold) {
      std::ostringstream oss;
      const auto& v = cut_pts[split_index];
      if (!std::isfinite(threshold)) {
        // According to IEEE 754, the result of comparison [lhs] < infinity
        // must be identical for all finite [lhs]. Same goes for operator >.
        oss << (semantic::CompareWithOp(0.0, op, threshold) ? "1" : "0");
      } else {
        auto loc = common::binary_search(v.begin(), v.end(), threshold);
        CHECK(loc != v.end());
        oss << "data[" << split_index << "].qvalue " << semantic::OpName(op)
            << " " << static_cast<size_t>(loc - v.begin()) * 2;
      }
      return oss.str();
    };
  }
  std::vector<std::string> CommonHeader() const {
    return {"",
            "union Entry {",
            "  int missing;",
            "  float fvalue;",
            "  int qvalue;",
            "};"};
  }
  std::vector<std::string> ConstantsPreamble() const {
    std::vector<std::string> ret;
    ret.emplace_back("static const unsigned char is_categorical[] = {");
    {
      std::ostringstream oss, oss2;
      size_t length = 2;
      oss << "  ";
      const int num_feature = GetInfo().num_feature;
      const auto& is_categorical = GetInfo().is_categorical;
      for (int fid = 0; fid < num_feature; ++fid) {
        if (is_categorical[fid]) {
          common::WrapText(&oss, &length, "1", 80);
        } else {
          common::WrapText(&oss, &length, "0", 80);
        }
      }
      ret.push_back(oss.str());
      ret.emplace_back("};");
      ret.emplace_back();
    }
    ret.emplace_back("static const float threshold[] = {");
    {
      std::ostringstream oss, oss2;
      size_t length = 2;
      oss << "  ";
      for (const auto& e : GetInfo().cut_pts) {
        for (const auto& value : e) {
          oss2.clear(); oss2.str(std::string()); oss2 << value;
          common::WrapText(&oss, &length, oss2.str(), 80);
        }
      }
      ret.push_back(oss.str());
      ret.emplace_back("};");
      ret.emplace_back();
    }
    ret.emplace_back("static const int th_begin[] = {");
    {
      std::ostringstream oss, oss2;
      size_t length = 2;
      size_t accum = 0;
      oss << "  ";
      for (const auto& e : GetInfo().cut_pts) {
        oss2.clear(); oss2.str(std::string()); oss2 << accum;
        common::WrapText(&oss, &length, oss2.str(), 80);
        accum += e.size();
      }
      ret.push_back(oss.str());
      ret.emplace_back("};");
      ret.emplace_back();
    }
    ret.emplace_back("static const int th_len[] = {");
    {
      std::ostringstream oss, oss2;
      size_t length = 2;
      oss << "  ";
      for (const auto& e : GetInfo().cut_pts) {
        oss2.clear(); oss2.str(std::string()); oss2 << e.size();
        common::WrapText(&oss, &length, oss2.str(), 80);
      }
      ret.push_back(oss.str());
      ret.emplace_back("};");
      ret.emplace_back();
    }

    auto func = semantic::FunctionBlock(
        "static inline int quantize(float val, unsigned fid)",
        semantic::PlainBlock(
           {"const float* array = &threshold[th_begin[fid]];",
            "int len = th_len[fid];",
            "int low = 0;",
            "int high = len;",
            "int mid;",
            "float mval;",
            "if (val < array[0]) {",
            "  return -10;",
            "}",
            "while (low + 1 < high) {",
            "  mid = (low + high) / 2;",
            "  mval = array[mid];",
            "  if (val == mval) {",
            "    return mid * 2;",
            "  } else if (val < mval) {",
            "    high = mid;",
            "  } else {",
            "    low = mid;",
            "  }",
            "}",
            "if (array[low] == val) {",
            "  return low * 2;",
            "} else if (high == len) {",
            "  return len * 2;",
            "} else {",
            "  return low * 2 + 1;",
            "}"}), nullptr).Compile();
    ret.insert(ret.end(), func.begin(), func.end());
    return ret;
  }
  std::vector<std::string> Preprocessing() const {
    return quant_preamble;
  }
  bool QuantizeFlag() const {
    return true;
  }
 private:
  std::vector<std::string> quant_preamble;
};

inline std::vector<std::vector<tl_float>>
ExtractCutPoints(const Model& model) {
  std::vector<std::vector<tl_float>> cut_pts;

  std::vector<std::set<tl_float>> thresh_;
  cut_pts.resize(model.num_feature);
  thresh_.resize(model.num_feature);
  for (size_t i = 0; i < model.trees.size(); ++i) {
    const Tree& tree = model.trees[i];
    std::queue<int> Q;
    Q.push(0);
    while (!Q.empty()) {
      const int nid = Q.front();
      const Tree::Node& node = tree[nid];
      Q.pop();
      if (!node.is_leaf()) {
        if (node.split_type() == SplitFeatureType::kNumerical) {
          const tl_float threshold = node.threshold();
          const unsigned split_index = node.split_index();
          if (std::isfinite(threshold)) {  // ignore infinity
            thresh_[split_index].insert(threshold);
          }
        } else {
          CHECK(node.split_type() == SplitFeatureType::kCategorical);
        }
        Q.push(node.cleft());
        Q.push(node.cright());
      }
    }
  }
  for (int i = 0; i < model.num_feature; ++i) {
    std::copy(thresh_[i].begin(), thresh_[i].end(),
              std::back_inserter(cut_pts[i]));
  }
  return cut_pts;
}

TREELITE_REGISTER_COMPILER(RecursiveCompiler, "recursive")
.describe("A compiler with a recursive approach")
.set_body([](const CompilerParam& param) -> Compiler* {
    if (param.quantize > 0) {
      return new RecursiveCompiler<Quantize>(param);
    } else {
      return new RecursiveCompiler<NoQuantize>(param);
    }
  });
}  // namespace compiler
}  // namespace treelite

namespace {


inline void
GroupPolicy::Init(const treelite::Model& model) {
  this->num_output_group = model.num_output_group;
  this->random_forest_flag = model.random_forest_flag;
}

inline std::string
GroupPolicy::GroupQueryFunction() const {
  return "return " + std::to_string(num_output_group) + ";";
}

inline std::string
GroupPolicy::Accumulator() const {
  if (num_output_group > 1) {
    return std::string("float sum[") + std::to_string(num_output_group)
           + "] = {0.0f};\n  unsigned int tmp;";
  } else {
    return "float sum = 0.0f;\n  unsigned int tmp;";
  }
}

inline std::string
GroupPolicy::AccumulateTranslationUnit(size_t unit_id) const {
  if (num_output_group > 1) {
    return std::string("predict_margin_multiclass_unit")
         + std::to_string(unit_id) + "(data, sum);";
  } else {
    return std::string("sum += predict_margin_unit")
         + std::to_string(unit_id) + "(data);";
  }
}

inline std::vector<std::string>
GroupPolicy::AccumulateLeaf(const treelite::Tree::Node& node,
                            size_t tree_id) const {
  if (num_output_group > 1) {
    if (random_forest_flag) {
      // multi-class classification with random forest
      const std::vector<treelite::tl_float>& leaf_vector = node.leaf_vector();
      CHECK_EQ(leaf_vector.size(), static_cast<size_t>(num_output_group))
        << "Ill-formed model: leaf vector must be of length [num_output_group]";
      std::vector<std::string> lines;
      lines.reserve(num_output_group);
      for (int group_id = 0; group_id < num_output_group; ++group_id) {
        lines.push_back(std::string("sum[") + std::to_string(group_id)
          + "] += (float)"
          + treelite::common::ToString(leaf_vector[group_id]) + ";");
      }
      return lines;
    } else {
      // multi-class classification with gradient boosted trees
      const treelite::tl_float leaf_value = node.leaf_value();
      return { std::string("sum[") + std::to_string(tree_id % num_output_group)
             + "] += (float)" + treelite::common::ToString(leaf_value) + ";" };
    }
  } else {
    const treelite::tl_float leaf_value = node.leaf_value();
    return {std::string("sum += (float)")
            + treelite::common::ToString(leaf_value) + ";" };
  }
}

inline std::vector<std::string>
GroupPolicy::Return() const {
  if (num_output_group > 1) {
    return {std::string("for (int i = 0; i < ")
                + std::to_string(num_output_group) + "; ++i) {",
            "  result[i] += sum[i];",
            "}" };
  } else {
    return { "return sum;" };
  }
}

inline std::vector<std::string>
GroupPolicy::FinalReturn(size_t num_tree, float global_bias) const {
  if (num_output_group > 1) {
    if (random_forest_flag) {
      // multi-class classification with random forest
      return {std::string("for (int i = 0; i < ")
                + std::to_string(num_output_group) + "; ++i) {",
              std::string("  result[i] = sum[i] / ")
                  + std::to_string(num_tree) + " + ("
                  + treelite::common::ToString(global_bias) + ");",
              "}"};
    } else {
      // multi-class classification with gradient boosted trees
      return {std::string("for (int i = 0; i < ")
                  + std::to_string(num_output_group) + "; ++i) {",
              "  result[i] = sum[i] + ("
                  + treelite::common::ToString(global_bias) + ");",
              "}"};
    }
  } else {
    if (random_forest_flag) {
      return { std::string("return sum / ") + std::to_string(num_tree) + " + ("
                  + treelite::common::ToString(global_bias) + ");" };
    } else {
      return { std::string("return sum + (")
                  + treelite::common::ToString(global_bias) + ");" };
    }
  }
}

inline std::string
GroupPolicy::Prototype() const {
  if (num_output_group > 1) {
    return "void predict_margin_multiclass(union Entry* data, float* result)";
  } else {
    return "float predict_margin(union Entry* data)";
  }
}

inline std::string
GroupPolicy::PrototypeTranslationUnit(size_t unit_id) const {
  if (num_output_group > 1) {
    return std::string("void predict_margin_multiclass_unit")
         + std::to_string(unit_id) + "(union Entry* data, float* result)";
  } else {
    return std::string("float predict_margin_unit")
         + std::to_string(unit_id) + "(union Entry* data)";
  }
}

}  // namespace anonymous
