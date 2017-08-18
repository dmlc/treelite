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
#include "param.h"

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
    const std::string comp
      = std::string("((") + std::to_string(categorical_bitmap)
          + "U >> (unsigned int)(data[" + std::to_string(split_index)
          + "].fvalue)) & 1)";
    return ((default_left) ?  (std::string("!(") + bitmap + ") || ")
                            : (std::string(" (") + bitmap + ") && "))
            + ((categorical_bitmap == 0) ? std::string("0") : comp);
  }

 private:
  unsigned split_index;
  bool default_left;
  uint64_t categorical_bitmap;

  inline uint64_t to_bitmap(const std::vector<uint8_t>& left_categories) const {
    uint64_t result = 0;
    for (uint8_t e : left_categories) {
      CHECK_LT(e, 64) << "Cannot have more than 64 categories in a feature";
      result |= (static_cast<uint64_t>(1) << e);
    }
    return result;
  }
};

}  // namespace anonymous

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(recursive);

std::vector<std::vector<tl_float>> ExtractCutPoints(const Model& model);

struct Metadata {
  int num_features;
  std::vector<std::vector<tl_float>> cut_pts;
  std::vector<bool> is_categorical;

  inline void Init(const Model& model, bool extract_cut_pts = false) {
    num_features = model.num_features;
    is_categorical.clear();
    is_categorical.resize(num_features, false);
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
                  << param.annotate_in << "\"";
      }
    }

    SemanticModel semantic_model;
    SequenceBlock sequence;
    if (param.parallel_comp > 0) {
      if (param.verbose > 0) {
        LOG(INFO) << "Parallel compilation enabled; member trees will be "
                  << "grouped in " << param.parallel_comp << " groups.";
      }
      const size_t ngroup = param.parallel_comp;
      sequence.Reserve(model.trees.size() + 3);
      sequence.PushBack(PlainBlock("float sum = 0.0f;"));
      sequence.PushBack(PlainBlock(QuantizePolicy::Preprocessing()));
      for (size_t group_id = 0; group_id < ngroup; ++group_id) {
        sequence.PushBack(PlainBlock(std::string("sum += predict_margin_group")
                                     + std::to_string(group_id)
                                     + "(data);"));
      }
      sequence.PushBack(PlainBlock("return sum;"));
    } else {
      if (param.verbose > 0) {
        LOG(INFO) << "Parallel compilation disabled; all member trees will be "
                  << "dump to a single source file. This may increase "
                  << "compilation time and memory usage.";
      }
      sequence.Reserve(model.trees.size() + 3);
      sequence.PushBack(PlainBlock("float sum = 0.0f;"));
      sequence.PushBack(PlainBlock(QuantizePolicy::Preprocessing()));
      for (size_t tree_id = 0; tree_id < model.trees.size(); ++tree_id) {
        const Tree& tree = model.trees[tree_id];
        if (!annotation.empty()) {
          sequence.PushBack(common::MoveUniquePtr(WalkTree(tree,
                                                  annotation[tree_id])));
        } else {
          sequence.PushBack(common::MoveUniquePtr(WalkTree(tree, {})));
        }
      }
      sequence.PushBack(PlainBlock("return sum;"));
    }
    FunctionBlock function("float predict_margin(union Entry* data)",
      std::move(sequence), &semantic_model.function_registry, true);
    auto file_preamble = QuantizePolicy::PreprocessingPreamble();
    semantic_model.units.emplace_back(PlainBlock(file_preamble),
                                      std::move(function));
    if (param.parallel_comp > 0) {
      const size_t ngroup = param.parallel_comp;
      const size_t group_size = (model.trees.size() + ngroup - 1) / ngroup;
      for (size_t group_id = 0; group_id < ngroup; ++group_id) {
        const size_t tree_begin = group_id * group_size;
        const size_t tree_end = std::min((group_id + 1) * group_size,
                                         model.trees.size());
        SequenceBlock group_seq;
        group_seq.Reserve(tree_end - tree_begin + 2);
        group_seq.PushBack(PlainBlock("float sum = 0.0f;"));
        for (size_t tree_id = tree_begin; tree_id < tree_end; ++tree_id) {
          const Tree& tree = model.trees[tree_id];
          if (!annotation.empty()) {
            group_seq.PushBack(common::MoveUniquePtr(WalkTree(tree,
                                                     annotation[tree_id])));
          } else {
            group_seq.PushBack(common::MoveUniquePtr(WalkTree(tree, {})));
          }
        }
        group_seq.PushBack(PlainBlock("return sum;"));
        FunctionBlock group_func(std::string("float predict_margin_group")
                                 + std::to_string(group_id)
                                 + "(union Entry* data)", std::move(group_seq),
                                 &semantic_model.function_registry);
        semantic_model.units.emplace_back(PlainBlock(), std::move(group_func));
      }
    }
    auto header = QuantizePolicy::CommonHeader();
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

  std::unique_ptr<CodeBlock> WalkTree(const Tree& tree,
                                      const std::vector<size_t>& counts) const {
    return WalkTree_(tree, counts, 0);
  }

  std::unique_ptr<CodeBlock> WalkTree_(const Tree& tree,
                                       const std::vector<size_t>& counts,
                                       int nid) const {
    using semantic::BranchHint;
    const Tree::Node& node = tree[nid];
    if (node.is_leaf()) {
      const tl_float leaf_value = node.leaf_value();
      return std::unique_ptr<CodeBlock>(new PlainBlock(
        std::string("sum += (float)")
          + common::FloatToString(leaf_value) + ";"));
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
        common::MoveUniquePtr(WalkTree_(tree, counts, node.cleft())),
        common::MoveUniquePtr(WalkTree_(tree, counts, node.cright())),
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
      oss << "data[" << split_index << "].fvalue "
          << semantic::OpName(op) << " " << threshold;
      return oss.str();
    };
  }
  std::vector<std::string> CommonHeader() const {
    return {"union Entry {",
            "  int missing;",
            "  float fvalue;",
            "};"};
  }
  std::vector<std::string> PreprocessingPreamble() const {
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
      + std::to_string(GetInfo().num_features) + "; ++i) {",
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
      auto loc = common::binary_search(v.begin(), v.end(), threshold);
      CHECK(loc != v.end());
      oss << "data[" << split_index << "].qvalue " << semantic::OpName(op)
          << " " << static_cast<size_t>(loc - v.begin()) * 2;
      return oss.str();
    };
  }
  std::vector<std::string> CommonHeader() const {
    return {"union Entry {",
            "  int missing;",
            "  float fvalue;",
            "  int qvalue;",
            "};"};
  }
  std::vector<std::string> PreprocessingPreamble() const {
    std::vector<std::string> ret;
    ret.emplace_back("static const unsigned char is_categorical[] = {");
    {
      std::ostringstream oss, oss2;
      size_t length = 2;
      oss << "  ";
      const int num_features = GetInfo().num_features;
      const auto& is_categorical = GetInfo().is_categorical;
      for (int fid = 0; fid < num_features; ++fid) {
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
  cut_pts.resize(model.num_features);
  thresh_.resize(model.num_features);
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
          thresh_[split_index].insert(threshold);
          if (split_index == 0) {
            LOG(INFO) << "Inserting " << threshold << " into cut_pts[" << split_index << "]";
          }
        } else {
          CHECK(node.split_type() == SplitFeatureType::kCategorical);
        }
        Q.push(node.cleft());
        Q.push(node.cright());
      }
    }
  }
  for (int i = 0; i < model.num_features; ++i) {
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
