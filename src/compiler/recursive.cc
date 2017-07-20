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
#include <queue>
#include <algorithm>
#include <iterator>
#include "param.h"

namespace treelite {
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

template <typename Policy>
class RecursiveCompiler : public Compiler, private Policy {
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
    info.Init(model, Policy::QuantizeFlag());
    Policy::Init(std::move(info));

    SequenceBlock sequence;
    sequence.Reserve(model.trees.size() + 4);
    sequence.PushBack(PlainBlock("float sum = 0.0f;"));
    sequence.PushBack(PlainBlock(Policy::LocalVariables()));
    for (const auto& tree : model.trees) {
      sequence.PushBack(MoveUniquePtr(WalkTree(tree)));
    }
    sequence.PushBack(PlainBlock("return sum;"));

    FunctionBlock function(
      std::string("float predict_margin(") + Policy::Prototype() + ")",
      std::move(sequence));

    auto preamble = Policy::Preamble();
    preamble.emplace_back();
    auto file = common::make_unique<SequenceBlock>();
    file->Reserve(2);
    file->PushBack(PlainBlock(std::move(preamble)));
    file->PushBack(std::move(function));

    return file;
  }

 private:
  CompilerParam param;
  
  std::unique_ptr<CodeBlock> WalkTree(const Tree& tree) const {
    return WalkTree_(tree, 0);
  }

  std::unique_ptr<CodeBlock> WalkTree_(const Tree& tree, int nid) const {
    const Tree::Node& node = tree[nid];
    if (node.is_leaf()) {
      SimpleAccumulator sa("sum");
      const tl_float leaf_value = node.leaf_value();
      return std::unique_ptr<CodeBlock>(new PlainBlock(sa.Compile(leaf_value)));
    } else {
      return std::unique_ptr<CodeBlock>(new IfElseBlock(
        SplitCondition(node,
                       MoveUniquePtr(Policy::Feature()),
                       MoveUniquePtr(Policy::Numeric())),
        MoveUniquePtr(WalkTree_(tree, node.cleft())),
        MoveUniquePtr(WalkTree_(tree, node.cright()))
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

template <typename QuantizePolicy>
class DenseLayout : protected QuantizePolicy,
                    private virtual MetadataStore {
 protected:
  using DenseFeature = semantic::DenseFeature;
  template <typename... Args>
  void Init(Args&&... args) {
     MetadataStore::Init(std::forward<Args>(args)...);
     QuantizePolicy::Init([&] (const std::vector<std::string>& content) {
       return IterateOverFeatures(content);
     });
  }
  std::unique_ptr<DenseFeature> Feature() const {
    return common::make_unique<DenseFeature>("bitmap",
                                             QuantizePolicy::DataArrayName());
  }
  std::vector<std::string>
  IterateOverFeatures(const std::vector<std::string>& content) {
    std::vector<std::string> ret{
      std::string("for (int i = 0; i < ")
      + std::to_string(GetInfo().num_features) + "; ++i) {",
      "  if (bitmap[i]) {",
      "    const float val = data[i];",
      "    const int fid = i;"};
    semantic::TransformPushBack(&ret, content, [] (std::string line) {
      return "    " + line;
    });
    ret.push_back("  }");
    ret.push_back("}");
    return ret;
  }
  std::string Prototype() const {
    return "const unsigned char* bitmap, const float* data";
  }
  std::vector<std::string> Preamble() const {
    return QuantizePolicy::Preamble();
  }
  std::vector<std::string> LocalVariables() const {
    return QuantizePolicy::LocalVariables();
  }
};

template <typename QuantizePolicy>
class CompressedDenseLayout : protected QuantizePolicy,
                              private virtual MetadataStore {
 protected:
  using CompressedDenseFeature = semantic::CompressedDenseFeature;
  template <typename... Args>
  void Init(Args&&... args) {
     MetadataStore::Init(std::forward<Args>(args)...);
     QuantizePolicy::Init([&] (const std::vector<std::string>& content) {
       return IterateOverFeatures(content);
     });
  }
  std::unique_ptr<CompressedDenseFeature> Feature() const {
    return common::make_unique<CompressedDenseFeature>("bitmap", "ACCESS",
                                              QuantizePolicy::DataArrayName());
  }
  std::vector<std::string>
  IterateOverFeatures(const std::vector<std::string>& content) {
    std::vector<std::string> ret{
      std::string("for (int i = 0; i < ")
      + std::to_string(GetInfo().num_features) + "; ++i) {",
      "  if (ACCESS(bitmap, i / 8, i % 8)) {",
      "    const float val = data[i];",
      "    const int fid = i;"};
    semantic::TransformPushBack(&ret, content, [] (std::string line) {
      return "    " + line;
    });
    ret.push_back("  }");
    ret.push_back("}");
    return ret;
  }
  std::string Prototype() const {
    return "const unsigned char* bitmap, const float* data";
  }
  std::vector<std::string> Preamble() const {
    std::vector<std::string> ret{ 
       "#define ACCESS(x, i, offset) ((x[(i)] & (1 << offset)) >> offset)"};
    auto ret2 = QuantizePolicy::Preamble();
    ret.insert(ret.end(), ret2.begin(), ret2.end());
    return ret;
  }
  std::vector<std::string> LocalVariables() const {
    return QuantizePolicy::LocalVariables();
  }
};

template <typename QuantizePolicy>
class SparseLayout : protected QuantizePolicy,
                     private virtual MetadataStore {
 protected:
  using SparseFeature = semantic::SparseFeature;
  template <typename... Args>
  void Init(Args&&... args) {
     MetadataStore::Init(std::forward<Args>(args)...);
     QuantizePolicy::Init([&] (const std::vector<std::string>& content) {
       return IterateOverFeatures(content);
     });
  }
  std::unique_ptr<SparseFeature> Feature() const {
    return common::make_unique<SparseFeature>(QuantizePolicy::DataArrayName(),
                                              "len", "col_ind", "lookup");
  }
  std::vector<std::string>
  IterateOverFeatures(const std::vector<std::string>& content) {
    std::vector<std::string> ret{
      "for (int i = 0; i < len; ++i) {",
      "  const float val = data[i];",
      "  const int fid = col_ind[i];"};
    semantic::TransformPushBack(&ret, content, [] (std::string line) {
      return "  " + line;
    });
    ret.push_back("}");
    return ret;
  }
  std::string Prototype() const {
    return "const float* data, int len, const int* col_ind";
  }
  std::vector<std::string> Preamble() const {
    auto ret = semantic::FunctionBlock(
            "int lookup(const int* col_ind, int len, int offset)",
            semantic::PlainBlock({
            "int i;",
            "for (i = 0; i < len; ++i) {",
            "  if (col_ind[i] == offset) {",
            "    return i;",
            "  }",
            "}",
            "return -1;"})).Compile();
    auto ret2 = QuantizePolicy::Preamble();
    ret.insert(ret.end(), ret2.begin(), ret2.end());
    return ret;
  }
  std::vector<std::string> LocalVariables() const {
    std::vector<std::string> ret{"int idx;"};
    auto ret2 = QuantizePolicy::LocalVariables();
    ret.insert(ret.end(), ret2.begin(), ret2.end());
    return ret;
  }
};

class NoQuantize : protected virtual MetadataStore {
 protected:
  using SimpleNumeric = semantic::SimpleNumeric;
  template <typename Func>
  void Init(Func f) {}
  std::string DataArrayName() const {
    return "data";
  }
  std::unique_ptr<SimpleNumeric> Numeric() const {
    return common::make_unique<SimpleNumeric>();
  }
  std::vector<std::string> LocalVariables() const {
    return {};
  }
  std::vector<std::string> Preamble() const {
    return {};
  }
  bool QuantizeFlag() const {
    return false;
  }
};

class Quantize : protected virtual MetadataStore {
 protected:
  using QuantizeNumeric = semantic::QuantizeNumeric;
  template <typename Func>
  void Init(Func f) {
    quant_preamble = f({"quantized[i] = quantize(val, fid);"});
  }
  std::string DataArrayName() const {
    return "quantized";
  }
  std::unique_ptr<QuantizeNumeric> Numeric() const {
    return common::make_unique<QuantizeNumeric>(GetInfo().cut_pts);
  }
  std::vector<std::string> LocalVariables() const {
    std::vector<std::string> ret{
            "static int quantized["
            + std::to_string(GetInfo().num_features) + "];"};
    ret.insert(ret.end(), quant_preamble.begin(), quant_preamble.end());
    return ret;
  }
  std::vector<std::string> Preamble() const {
    std::vector<std::string> ret{"const float threshold[] = {"};
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

    auto func = semantic::FunctionBlock("int quantize(float val, unsigned fid)",
                                        semantic::PlainBlock(
           {"const float* array = &threshold[th_begin[fid]];",
            "int len = th_len[fid];",
            "int low = 0;",
            "int high = len;",
            "int mid;",
            "float mval;",
            "if (val < array[0]) {",
            "  return -1;",
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
    switch(param.data_layout) {
     case 0:
      if (param.quantize > 0) {
        return new RecursiveCompiler<DenseLayout<Quantize>>(param);
      } else {
        return new RecursiveCompiler<DenseLayout<NoQuantize>>(param);
      }
     case 1:
      if (param.quantize > 0) {
        return new RecursiveCompiler<CompressedDenseLayout<Quantize>>(param);
      } else {
        return new RecursiveCompiler<CompressedDenseLayout<NoQuantize>>(param);
      }
     case 2:
      if (param.quantize > 0) {
        return new RecursiveCompiler<SparseLayout<Quantize>>(param);
      } else {
        return new RecursiveCompiler<SparseLayout<NoQuantize>>(param);
      }
    }
    return nullptr;  // should never get here
  });
}  // namespace compiler
}  // namespace treelite
