/*!
 * Copyright 2017 by Contributors
 * \file simple.cc
 * \brief Bare-bones simple compiler
 * \author Philip Cho
 */

#include <treelite/common.h>
#include <treelite/compiler.h>
#include <treelite/tree.h>
#include <treelite/semantic.h>

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(simple);

struct Constants {
  int num_features;

  explicit Constants(const Model& model)
    : num_features(model.num_features) {}
};

template <typename LayoutPolicy, typename QuantizePolicy>
class SimpleCompiler : public Compiler, private LayoutPolicy, QuantizePolicy {
 public:
  SimpleCompiler() { LOG(INFO) << "SimpleCompiler yah"; }
  
  using CodeBlock = semantic::CodeBlock;
  using PlainBlock = semantic::PlainBlock;
  using FunctionBlock = semantic::FunctionBlock;
  using SequenceBlock = semantic::SequenceBlock;
  using SimpleAccumulator = semantic::SimpleAccumulator;
  using IfElseBlock = semantic::IfElseBlock;
  using SplitCondition = semantic::SplitCondition;
  using SimpleNumeric = semantic::SimpleNumeric;

  std::unique_ptr<CodeBlock>
  Export(const Model& model) const override {
    SequenceBlock sequence;
    sequence.Reserve(model.trees.size() + 3);
    sequence.PushBack(PlainBlock("float sum = 0.0f;"));
    sequence.PushBack(PlainBlock(LayoutPolicy::LocalVariables()));
    for (const auto& tree : model.trees) {
      sequence.PushBack(MoveUniquePtr(WalkTree(tree)));
    }
    sequence.PushBack(PlainBlock("return sum;"));

    FunctionBlock function(
      std::string("float predict_margin(") + LayoutPolicy::Prototype() + ")",
      std::move(sequence));

    Constants cts(model);
    auto preamble = LayoutPolicy::Preamble(cts);
    preamble.emplace_back();
    std::unique_ptr<SequenceBlock> file(new SequenceBlock);
    file->Reserve(2);
    file->PushBack(PlainBlock(std::move(preamble)));
    file->PushBack(std::move(function));

    return file;
  }

 private:
  std::unique_ptr<CodeBlock> WalkTree(const Tree& tree) const {
    return WalkTree_(tree, 0);
  }

  std::unique_ptr<CodeBlock> WalkTree_(const Tree& tree, int nid) const {
    const Tree::Node& node = tree[nid];
    if (node.is_leaf()) {
      SimpleAccumulator sa("sum", SimpleNumeric());
      const tl_float leaf_value = node.leaf_value();
      return std::unique_ptr<CodeBlock>(new PlainBlock(sa.Compile(leaf_value)));
    } else {
      return std::unique_ptr<CodeBlock>(new IfElseBlock(
        SplitCondition(node,
                       MoveUniquePtr(LayoutPolicy::Feature()),
                       MoveUniquePtr(QuantizePolicy::Numeric())),
        MoveUniquePtr(WalkTree_(tree, node.cleft())),
        MoveUniquePtr(WalkTree_(tree, node.cright()))
      ));
    }
  }
};

class DenseLayout {
 protected:
  using DenseFeature = semantic::DenseFeature;
  using CodeBlock = semantic::CodeBlock;
  std::unique_ptr<DenseFeature> Feature() const {
    return common::make_unique<DenseFeature>("bitmap", "data");
  }
  std::string Prototype() const {
    return "const unsigned char* bitmap, const float* data";
  }
  std::vector<std::string> Preamble(const Constants& ct) const {
    return {};
  }
  std::vector<std::string> LocalVariables() const {
    return {};
  }
};

class CompressedDenseLayout {
 protected:
  using CompressedDenseFeature = semantic::CompressedDenseFeature;
  using CodeBlock = semantic::CodeBlock;
  std::unique_ptr<CompressedDenseFeature> Feature() const {
    return common::make_unique<CompressedDenseFeature>("bitmap", "ACCESS",
                                                       "data");
  }
  std::string Prototype() const {
    return "const unsigned char* bitmap, const float* data";
  }
  std::vector<std::string> Preamble(const Constants& ct) const {
    return
      {"#define ACCESS(x, i, offset) ((x[(i)] & (1 << offset)) >> offset)"};
  }
  std::vector<std::string> LocalVariables() const {
    return {};
  }
};

class SparseLayout {
 protected:
  using SparseFeature = semantic::SparseFeature;
  using CodeBlock = semantic::CodeBlock;
  using PlainBlock = semantic::PlainBlock;
  std::unique_ptr<SparseFeature> Feature() const {
    return common::make_unique<SparseFeature>("data", "col_ind", "lookup");
  }
  std::string Prototype() const {
    return "const float* data, const int* col_ind";
  }
  std::vector<std::string> Preamble(const Constants& ct) const {
    return {"int lookup(const int* col_ind, int offset) {",
            "  int i;",
            std::string("  for (i = 0; i < ")
            + std::to_string(ct.num_features) + "; ++i) {",
            "    if (col_ind[i] == offset) {",
            "      return i;",
            "    }",
            "  }",
            "  return -1;",
            "}"};
  }
  std::vector<std::string> LocalVariables() const {
    return {"int idx;"};
  }
};

class NoQuantize {
 protected:
  using SimpleNumeric = semantic::SimpleNumeric;
  std::unique_ptr<SimpleNumeric> Numeric() const {
    return std::unique_ptr<SimpleNumeric>(new SimpleNumeric);
  }
};

TREELITE_REGISTER_COMPILER(SimpleCompiler, "simple")
.describe("Bare-bones simple compiler")
.set_body([]() {
    return new SimpleCompiler<DenseLayout, NoQuantize>();
  });
TREELITE_REGISTER_COMPILER(SimpleCompilerCompressedDenseLayout, "compressed")
.describe("Bare-bones simple compiler (compressed bitmap)")
.set_body([]() {
    return new SimpleCompiler<CompressedDenseLayout, NoQuantize>();
  });
TREELITE_REGISTER_COMPILER(SimpleCompilerSparseLayout, "sparse")
.describe("Bare-bones simple compiler (sparse row)")
.set_body([]() {
    return new SimpleCompiler<SparseLayout, NoQuantize>();
  });
}  // namespace compiler
}  // namespace treelite
