/*!
 * Copyright 2017 by Contributors
 * \file recursive.cc
 * \brief Recursive compiler
 * \author Philip Cho
 */

#include <treelite/common.h>
#include <treelite/compiler.h>
#include <treelite/tree.h>
#include <treelite/semantic.h>
#include <dmlc/data.h>
#include <dmlc/json.h>
#include <queue>
#include <algorithm>
#include <iterator>
#include "param.h"

namespace treelite {

namespace {
  using Annotation = std::vector<std::vector<size_t>>;
}  // namespace anonymous

namespace compiler {

DMLC_REGISTRY_FILE_TAG(recursive);

std::vector<std::vector<tl_float>> ExtractCutPoints(const Model& model);

struct Metadata {
  int num_features;
  std::vector<std::vector<tl_float>> cut_pts;

  inline void Init(const Model& model, bool extract_cut_pts = false) {
    num_features = model.num_features;
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
    LOG(INFO) << "RecursiveCompiler yah";
  }
  
  using CodeBlock = semantic::CodeBlock;
  using PlainBlock = semantic::PlainBlock;
  using FunctionBlock = semantic::FunctionBlock;
  using SequenceBlock = semantic::SequenceBlock;
  using SimpleAccumulator = semantic::SimpleAccumulator;
  using IfElseBlock = semantic::IfElseBlock;
  using SplitCondition = semantic::SplitCondition;
  using SimpleNumeric = semantic::SimpleNumeric;

  std::unique_ptr<CodeBlock>
  Export(const Model& model) override {
    Metadata info;
    info.Init(model, QuantizePolicy::QuantizeFlag());
    QuantizePolicy::Init(std::move(info));

    Annotation annotation;
    bool annotate = false;
    if (param.annotate_in != "NULL") {
      std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(
                                       param.annotate_in.c_str(), "r"));
      dmlc::istream is(fi.get());
      auto reader = common::make_unique<dmlc::JSONReader>(&is);
      reader->Read(&annotation);
      annotate = true;
    }

    SequenceBlock sequence;
    sequence.Reserve(model.trees.size() + 4);
    sequence.PushBack(PlainBlock("float sum = 0.0f;"));
    for (size_t tree_id = 0; tree_id < model.trees.size(); ++tree_id) {
      const Tree& tree = model.trees[tree_id];
      if (!annotation.empty()) {
        sequence.PushBack(MoveUniquePtr(WalkTree(tree, annotation[tree_id])));
      } else {
        sequence.PushBack(MoveUniquePtr(WalkTree(tree, {})));
      }
    }
    sequence.PushBack(PlainBlock("return sum;"));

    FunctionBlock function("float predict_margin(const union Entry* data)",
      std::move(sequence));

    auto preamble = QuantizePolicy::Preamble();
    preamble.emplace_back();
    if (annotate) {
      preamble.emplace_back("#define LIKELY(x)     __builtin_expect(!!(x), 1)");
      preamble.emplace_back("#define UNLIKELY(x)   __builtin_expect(!!(x), 0)");
    }

    auto file = common::make_unique<SequenceBlock>();
    file->Reserve(2);
    file->PushBack(PlainBlock(std::move(preamble)));
    file->PushBack(std::move(function));

    return std::move(file);
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
    using semantic::LikelyDirection;
    const Tree::Node& node = tree[nid];
    if (node.is_leaf()) {
      SimpleAccumulator sa("sum");
      const tl_float leaf_value = node.leaf_value();
      return std::unique_ptr<CodeBlock>(new PlainBlock(sa.Compile(leaf_value)));
    } else {
      LikelyDirection likely_direction = LikelyDirection::kNone;
      if (!counts.empty()) {
        const size_t left_count = counts[node.cleft()];
        const size_t right_count = counts[node.cright()];
        likely_direction = (left_count > right_count) ? LikelyDirection::kLeft
                                                      : LikelyDirection::kRight;
      }
      return std::unique_ptr<CodeBlock>(new IfElseBlock(
        SplitCondition(node,
                       MoveUniquePtr(QuantizePolicy::Numeric()),
                       likely_direction),
        MoveUniquePtr(WalkTree_(tree, counts, node.cleft())),
        MoveUniquePtr(WalkTree_(tree, counts, node.cright()))
      ));
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
  using SimpleNumeric = semantic::SimpleNumeric;
  template <typename... Args>
  void Init(Args&&... args) {
    MetadataStore::Init(std::forward<Args>(args)...);
  }
  std::unique_ptr<SimpleNumeric> Numeric() const {
    return common::make_unique<SimpleNumeric>();
  }
  std::vector<std::string> Preamble() const {
    return {"union Entry {",
            "  int missing;",
            "  float fvalue;",
            "};"};
  }
  bool QuantizeFlag() const {
    return false;
  }
};

class Quantize : private MetadataStore {
 protected:
  using QuantizeNumeric = semantic::QuantizeNumeric;
  template <typename... Args>
  void Init(Args&&... args) {
    MetadataStore::Init(std::forward<Args>(args)...);
    quant_preamble = {
      std::string("for (int i = 0; i < ")
      + std::to_string(GetInfo().num_features) + "; ++i) {",
      "  if (data[i].missing != -1) {",
      "    data[i].qvalue = quantize(data[i].fvalue, i);",
      "  }",
      "}"};
  }
  std::unique_ptr<QuantizeNumeric> Numeric() const {
    return common::make_unique<QuantizeNumeric>(GetInfo().cut_pts);
  }
  std::vector<std::string> Preamble() const {
    std::vector<std::string> ret{"union Entry {",
                                 "  int missing;",
                                 "  float fvalue;",
                                 "  int qvalue;",
                                 "};"};
    ret.emplace_back("const float threshold[] = {");
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
    }
    ret.emplace_back("const int th_begin[] = {");
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
    }
    ret.emplace_back("const int th_len[] = {");
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
    }

    auto func = semantic::FunctionBlock("static inline int quantize(float val, unsigned fid)",
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
            "}"})).Compile();
    ret.insert(ret.end(), func.begin(), func.end());
    return ret;
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
        const tl_float threshold = node.threshold();
        const unsigned split_index = node.split_index();
        thresh_[split_index].insert(threshold);
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
