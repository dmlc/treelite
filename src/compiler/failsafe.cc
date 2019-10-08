/*!
 * Copyright (c) 2019 by Contributors
 * \file failsafe.cc
 * \author Philip Cho
 * \brief C code generator (fail-safe). The generated code will mimic prediction
 * logic found in XGBoost
 */

#include <treelite/tree.h>
#include <treelite/compiler.h>
#include <treelite/common.h>
#include <fmt/format.h>
#include <cmath>
#include <unordered_map>
#include <set>
#include <tuple>
#include <utility>
#include "./param.h"
#include "./pred_transform.h"
#include "./elf/elf_formatter.h"

#if defined(_MSC_VER) || defined(_WIN32)
#define DLLEXPORT_KEYWORD "__declspec(dllexport) "
#else
#define DLLEXPORT_KEYWORD ""
#endif

using namespace fmt::literals;

namespace {

struct NodeStructValue {
  unsigned int sindex;
  float info;
  int cleft;
  int cright;
};

const char* header_template = R"TREELITETEMPLATE(
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

union Entry {{
  int missing;
  float fvalue;
}};

union NodeInfo {{
  float leaf_value;
  float threshold;
}};

struct Node {{
  unsigned int sindex;
  union NodeInfo info;
  int cleft;
  int cright;
}};

extern const struct Node nodes[];
extern const int nodes_row_ptr[];

{dllexport}size_t get_num_output_group(void);
{dllexport}size_t get_num_feature(void);
{dllexport}{predict_function_signature};
)TREELITETEMPLATE";

const char* main_template = R"TREELITETEMPLATE(
#include "header.h"

{nodes_row_ptr}

size_t get_num_output_group(void) {{
  return {num_output_group};
}}

size_t get_num_feature(void) {{
  return {num_feature};
}}

{pred_transform_function}

{predict_function_signature} {{
  {accumulator_definition};

  for (int tree_id = 0; tree_id < {num_tree}; ++tree_id) {{
    int nid = 0;
    const struct Node* tree = &nodes[nodes_row_ptr[tree_id]];
    while (tree[nid].cleft != -1) {{
      const unsigned feature_id = tree[nid].sindex & ((1U << 31) - 1U);
      const unsigned char default_left = (tree[nid].sindex >> 31) != 0;
      if (data[feature_id].missing == -1) {{
        nid = (default_left ? tree[nid].cleft : tree[nid].cright);
      }} else {{
        nid = (data[feature_id].fvalue {compare_op} tree[nid].info.threshold
               ? tree[nid].cleft : tree[nid].cright);
      }}
    }}
    {output_statement}
  }}
  {return_statement}
}}
)TREELITETEMPLATE";

const char* return_multiclass_template =
R"TREELITETEMPLATE(
  for (int i = 0; i < {num_output_group}; ++i) {{
    result[i] = sum[i] + (float)({global_bias});
  }}
  if (!pred_margin) {{
    return pred_transform(result);
  }} else {{
    return {num_output_group};
  }}
)TREELITETEMPLATE";  // only for multiclass classification

const char* return_template =
R"TREELITETEMPLATE(
  sum += (float)({global_bias});
  if (!pred_margin) {{
    return pred_transform(sum);
  }} else {{
    return sum;
  }}
)TREELITETEMPLATE";

const char* arrays_template = R"TREELITETEMPLATE(
#include "header.h"

{nodes}
)TREELITETEMPLATE";

// Returns formatted nodes[] and nodes_row_ptr[] arrays
// nodes[]: stores nodes from all decision trees
// nodes_row_ptr[]: marks bounaries between decision trees. The nodes belonging to Tree [i] are
//                  found in nodes[nodes_row_ptr[i]:nodes_row_ptr[i+1]]
inline std::pair<std::string, std::string> FormatNodesArray(const treelite::Model& model) {
  treelite::common::ArrayFormatter nodes(100, 2);
  treelite::common::ArrayFormatter nodes_row_ptr(100, 2);
  int node_count = 0;
  nodes_row_ptr << "0";
  for (const auto& tree : model.trees) {
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      const auto& node = tree[nid];
      if (node.is_leaf()) {
        CHECK(!node.has_leaf_vector())
          << "multi-class random forest classifier is not supported in FailSafeCompiler";
        nodes << fmt::format("{{ 0x{sindex:X}, {info}, {cleft}, {cright} }}",
          "sindex"_a = 0,
          "info"_a = node.leaf_value().ToString(),
          "cleft"_a = -1,
          "cright"_a = -1);
      } else {
        CHECK(node.split_type() == treelite::SplitFeatureType::kNumerical
              && node.left_categories().empty())
          << "categorical splits are not supported in FailSafeCompiler";
        nodes << fmt::format("{{ 0x{sindex:X}, {info}, {cleft}, {cright} }}",
          "sindex"_a = (node.split_index() | (static_cast<uint32_t>(node.default_left()) << 31)),
          "info"_a = node.threshold().ToString(),
          "cleft"_a = node.cleft(),
          "cright"_a = node.cright());
      }
    }
    node_count += tree.num_nodes;
    nodes_row_ptr << std::to_string(node_count);
  }
  return std::make_pair(fmt::format("const struct Node nodes[] = {{\n{}\n}};", nodes.str()),
                        fmt::format("const int nodes_row_ptr[] = {{\n{}\n}};",
                                    nodes_row_ptr.str()));
}

// Variant of FormatNodesArray(), where nodes[] array is dumped as an ELF binary
inline std::pair<std::vector<char>, std::string> FormatNodesArrayELF(const treelite::Model& model) {
  std::vector<char> nodes_elf;
  treelite::compiler::AllocateELFHeader(&nodes_elf);

  treelite::common::ArrayFormatter nodes_row_ptr(100, 2);
  NodeStructValue val;
  int node_count = 0;
  nodes_row_ptr << "0";
  for (const auto& tree : model.trees) {
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      const auto& node = tree[nid];
      if (node.is_leaf()) {
        CHECK(!node.has_leaf_vector())
          << "multi-class random forest classifier is not supported in FailSafeCompiler";
        const auto& leaf_value = node.leaf_value();
        CHECK(treelite::ADT::IsA<treelite::ADT::Float32Value>(leaf_value))
          << "FailSafeCompiler supports only models with float32 leaf values";
        val = {0, treelite::ADT::get<const treelite::ADT::Float32Value>(leaf_value), -1, -1};
      } else {
        CHECK(node.split_type() == treelite::SplitFeatureType::kNumerical
              && node.left_categories().empty())
          << "categorical splits are not supported in FailSafeCompiler";
        const auto& threshold = node.threshold();
        CHECK(treelite::ADT::IsA<treelite::ADT::Float32Value>(threshold))
          << "FailSafeCompiler supports only models with float32 split thresholds";
        val = {(node.split_index() | (static_cast<uint32_t>(node.default_left()) << 31)),
               treelite::ADT::get<const treelite::ADT::Float32Value>(threshold),
               node.cleft(), node.cright()};
      }
      const size_t beg = nodes_elf.size();
      nodes_elf.resize(beg + sizeof(NodeStructValue));
      std::memcpy(&nodes_elf[beg], &val, sizeof(NodeStructValue));
    }
    node_count += tree.num_nodes;
    nodes_row_ptr << std::to_string(node_count);
  }
  treelite::compiler::FormatArrayAsELF(&nodes_elf);

  return std::make_pair(nodes_elf, fmt::format("const int nodes_row_ptr[] = {{\n{}\n}};",
                                               nodes_row_ptr.str()));
}

// Get the comparison op used in the tree ensemble model
// If splits have more than one op, throw an error
inline std::string GetCommonOp(const treelite::Model& model) {
  std::set<treelite::Operator> ops;
  for (const auto& tree : model.trees) {
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      const auto& node = tree[nid];
      if (!node.is_leaf()) {
        ops.insert(node.comparison_op());
      }
    }
  }
  // sanity check: all numerical splits must have identical comparison operators
  CHECK_EQ(ops.size(), 1)
    << "FailSafeCompiler only supports models where all splits use identical comparison operator.";
  return treelite::OpName(*ops.begin());
}


// Test whether a string ends with a given suffix
inline bool EndsWith(const std::string& str, const std::string& suffix) {
  return (str.size() >= suffix.size()
          && str.compare(str.length() - suffix.size(), suffix.size(), suffix) == 0);
}

}   // anonymous namespace

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(failsafe);

class FailSafeCompiler : public Compiler {
 public:
  explicit FailSafeCompiler(const CompilerParam& param)
    : param(param) {
    if (param.verbose > 0) {
      LOG(INFO) << "Using FailSafeCompiler";
    }
    if (param.annotate_in != "NULL") {
      LOG(INFO) << "Warning: 'annotate_in' parameter is not applicable for "
                   "FailSafeCompiler";
    }
    if (param.quantize > 0) {
      LOG(INFO) << "Warning: 'quantize' parameter is not applicable for "
                   "FailSafeCompiler";
    }
    if (param.parallel_comp > 0) {
      LOG(INFO) << "Warning: 'parallel_comp' parameter is not applicable for "
                   "FailSafeCompiler";
    }
    if (std::isfinite(param.code_folding_req)) {
      LOG(INFO) << "Warning: 'code_folding_req' parameter is not applicable "
                   "for FailSafeCompiler";
    }
  }

  CompiledModel Compile(const Model& model) override {
    CompiledModel cm;
    cm.backend = "native";

    num_feature_ = model.num_feature;
    num_output_group_ = model.num_output_group;
    CHECK(!model.random_forest_flag)
      << "Only gradient boosted trees supported in FailSafeCompiler";
    pred_tranform_func_ = PredTransformFunction("native", model);
    files_.clear();

    const char* predict_function_signature
      = (num_output_group_ > 1) ?
          "size_t predict_multiclass(union Entry* data, int pred_margin, "
                                    "float* result)"
        : "float predict(union Entry* data, int pred_margin)";

    std::ostringstream main_program;
    std::string accumulator_definition
      = (num_output_group_ > 1
         ? fmt::format("float sum[{num_output_group}] = {{0.0f}}",
             "num_output_group"_a = num_output_group_)
         : std::string("float sum = 0.0f"));

    std::string output_statement
      = (num_output_group_ > 1
         ? fmt::format("sum[tree_id % {num_output_group}] += tree[nid].info.leaf_value;",
             "num_output_group"_a = num_output_group_)
         : std::string("sum += tree[nid].info.leaf_value;"));

    std::string return_statement
      = (num_output_group_ > 1
         ? fmt::format(return_multiclass_template,
             "num_output_group"_a = num_output_group_,
             "global_bias"_a = common::ToStringHighPrecision(model.param.global_bias))
         : fmt::format(return_template,
             "global_bias"_a = common::ToStringHighPrecision(model.param.global_bias)));

    std::string nodes, nodes_row_ptr;
    std::vector<char> nodes_elf;
    if (param.dump_array_as_elf > 0) {
      if (param.verbose > 0) {
        LOG(INFO) << "Dumping arrays as an ELF relocatable object...";
      }
      std::tie(nodes_elf, nodes_row_ptr) = FormatNodesArrayELF(model);
    } else {
      std::tie(nodes, nodes_row_ptr) = FormatNodesArray(model);
    }

    main_program << fmt::format(main_template,
      "nodes_row_ptr"_a = nodes_row_ptr,
      "pred_transform_function"_a = pred_tranform_func_,
      "predict_function_signature"_a = predict_function_signature,
      "num_output_group"_a = num_output_group_,
      "num_feature"_a = num_feature_,
      "num_tree"_a = model.trees.size(),
      "compare_op"_a = GetCommonOp(model),
      "accumulator_definition"_a = accumulator_definition,
      "output_statement"_a = output_statement,
      "return_statement"_a = return_statement);

    files_["main.c"] = CompiledModel::FileEntry(main_program.str());

    if (param.dump_array_as_elf > 0) {
      files_["arrays.o"] = CompiledModel::FileEntry(std::move(nodes_elf));
    } else {
      files_["arrays.c"] = CompiledModel::FileEntry(fmt::format(arrays_template,
        "nodes"_a = nodes));
    }

    files_["header.h"] = CompiledModel::FileEntry(fmt::format(header_template,
      "dllexport"_a = DLLEXPORT_KEYWORD,
      "predict_function_signature"_a = predict_function_signature));

    {
      /* write recipe.json */
      std::vector<std::unordered_map<std::string, std::string>> source_list;
      std::vector<std::string> extra_file_list;
      for (const auto& kv : files_) {
        if (EndsWith(kv.first, ".c")) {
          const size_t line_count
            = std::count(kv.second.content.begin(), kv.second.content.end(), '\n');
          source_list.push_back({ {"name",
                                   kv.first.substr(0, kv.first.length() - 2)},
                                  {"length", std::to_string(line_count)} });
        } else if (EndsWith(kv.first, ".o")) {
          extra_file_list.push_back(kv.first);
        }
      }
      std::ostringstream oss;
      auto writer = common::make_unique<dmlc::JSONWriter>(&oss);
      writer->BeginObject();
      writer->WriteObjectKeyValue("target", param.native_lib_name);
      writer->WriteObjectKeyValue("sources", source_list);
      if (!extra_file_list.empty()) {
        writer->WriteObjectKeyValue("extra", extra_file_list);
      }
      writer->EndObject();
      files_["recipe.json"] = CompiledModel::FileEntry(oss.str());
    }
    cm.files = std::move(files_);
    return cm;
  }

 private:
  CompilerParam param;
  int num_feature_;
  int num_output_group_;
  std::string pred_tranform_func_;
  std::unordered_map<std::string, CompiledModel::FileEntry> files_;
};

TREELITE_REGISTER_COMPILER(FailSafeCompiler, "failsafe")
.describe("Simple compiler to express trees as a tight for-loop")
.set_body([](const CompilerParam& param) -> Compiler* {
    return new FailSafeCompiler(param);
  });
}  // namespace compiler
}  // namespace treelite
