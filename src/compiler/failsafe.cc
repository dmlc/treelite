/*!
 * Copyright (c) 2019-2021 by Contributors
 * \file failsafe.cc
 * \brief C code generator (fail-safe). The generated code will mimic prediction logic found in
 *        XGBoost
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <treelite/logging.h>
#include <fmt/format.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <unordered_map>
#include <set>
#include <tuple>
#include <utility>
#include <cmath>
#include "./failsafe.h"
#include "./pred_transform.h"
#include "./common/format_util.h"
#include "./elf/elf_formatter.h"
#include "./native/main_template.h"
#include "./native/header_template.h"

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

const char* const header_template = R"TREELITETEMPLATE(
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

{query_functions_prototype}
{dllexport}{predict_function_signature};
)TREELITETEMPLATE";

const char* const main_template = R"TREELITETEMPLATE(
#include "header.h"

{nodes_row_ptr}

{query_functions_definition}

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

const char* const return_multiclass_template =
R"TREELITETEMPLATE(
  for (int i = 0; i < {num_class}; ++i) {{
    result[i] = sum[i] + (float)({global_bias});
  }}
  if (!pred_margin) {{
    return pred_transform(result);
  }} else {{
    return {num_class};
  }}
)TREELITETEMPLATE";  // only for multiclass classification

const char* const return_template =
R"TREELITETEMPLATE(
  sum += (float)({global_bias});
  if (!pred_margin) {{
    return pred_transform(sum);
  }} else {{
    return sum;
  }}
)TREELITETEMPLATE";

const char* const arrays_template = R"TREELITETEMPLATE(
#include "header.h"

{nodes}
)TREELITETEMPLATE";

// Returns formatted nodes[] and nodes_row_ptr[] arrays
// nodes[]: stores nodes from all decision trees
// nodes_row_ptr[]: marks bounaries between decision trees. The nodes belonging to Tree [i] are
//                  found in nodes[nodes_row_ptr[i]:nodes_row_ptr[i+1]]
inline std::pair<std::string, std::string> FormatNodesArray(
    const treelite::ModelImpl<float, float>& model) {
  treelite::compiler::common_util::ArrayFormatter nodes(100, 2);
  treelite::compiler::common_util::ArrayFormatter nodes_row_ptr(100, 2);
  int node_count = 0;
  nodes_row_ptr << "0";
  for (const auto& tree : model.trees) {
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      if (tree.IsLeaf(nid)) {
        TREELITE_CHECK(!tree.HasLeafVector(nid))
          << "multi-class random forest classifier is not supported in FailSafeCompiler";
        nodes << fmt::format("{{ 0x{sindex:X}, {info}, {cleft}, {cright} }}",
          "sindex"_a = 0,
          "info"_a = treelite::compiler::common_util::ToStringHighPrecision(tree.LeafValue(nid)),
          "cleft"_a = -1,
          "cright"_a = -1);
      } else {
        TREELITE_CHECK(tree.SplitType(nid) == treelite::SplitFeatureType::kNumerical
                       && tree.MatchingCategories(nid).empty())
          << "categorical splits are not supported in FailSafeCompiler";
        nodes << fmt::format("{{ 0x{sindex:X}, {info}, {cleft}, {cright} }}",
            "sindex"_a
              = (tree.SplitIndex(nid) |(static_cast<uint32_t>(tree.DefaultLeft(nid)) << 31U)),
            "info"_a = treelite::compiler::common_util::ToStringHighPrecision(tree.Threshold(nid)),
            "cleft"_a = tree.LeftChild(nid),
            "cright"_a = tree.RightChild(nid));
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
inline std::pair<std::vector<char>, std::string> FormatNodesArrayELF(
    const treelite::ModelImpl<float, float>& model) {
  std::vector<char> nodes_elf;
  treelite::compiler::AllocateELFHeader(&nodes_elf);

  treelite::compiler::common_util::ArrayFormatter nodes_row_ptr(100, 2);
  NodeStructValue val;
  int node_count = 0;
  nodes_row_ptr << "0";
  for (const auto& tree : model.trees) {
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      if (tree.IsLeaf(nid)) {
        TREELITE_CHECK(!tree.HasLeafVector(nid))
          << "multi-class random forest classifier is not supported in FailSafeCompiler";
        val = {0, static_cast<float>(tree.LeafValue(nid)), -1, -1};
      } else {
        TREELITE_CHECK(tree.SplitType(nid) == treelite::SplitFeatureType::kNumerical
                       && tree.MatchingCategories(nid).empty())
          << "categorical splits are not supported in FailSafeCompiler";
        val = {(tree.SplitIndex(nid) | (static_cast<uint32_t>(tree.DefaultLeft(nid)) << 31)),
               static_cast<float>(tree.Threshold(nid)), tree.LeftChild(nid), tree.RightChild(nid)};
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
inline std::string GetCommonOp(const treelite::ModelImpl<float, float>& model) {
  std::set<treelite::Operator> ops;
  for (const auto& tree : model.trees) {
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      if (!tree.IsLeaf(nid)) {
        ops.insert(tree.ComparisonOp(nid));
      }
    }
  }
  // sanity check: all numerical splits must have identical comparison operators
  TREELITE_CHECK_EQ(ops.size(), 1)
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

class FailSafeCompilerImpl {
 public:
  explicit FailSafeCompilerImpl(const CompilerParam& param) : param_(param) {}

  CompiledModel Compile(const Model& model_ptr) {
    TREELITE_CHECK(model_ptr.GetThresholdType() == TypeInfo::kFloat32
                   && model_ptr.GetLeafOutputType() == TypeInfo::kFloat32)
      << "Failsafe compiler only supports models with float32 thresholds and float32 leaf outputs";
    const auto& model = dynamic_cast<const ModelImpl<float, float>&>(model_ptr);

    CompiledModel cm;
    cm.backend = "native";

    num_feature_ = model.num_feature;
    num_class_ = model.task_param.num_class;
    TREELITE_CHECK(!model.average_tree_output)
      << "Averaging tree output is not supported in FailSafeCompiler";
    TREELITE_CHECK(model.task_type == TaskType::kBinaryClfRegr
                   || model.task_type == TaskType::kMultiClfGrovePerClass)
      << "Model task type unsupported by FailSafeCompiler";
    TREELITE_CHECK_EQ(model.task_param.leaf_vector_size, 1)
      << "Model with leaf vectors is not support by FailSafeCompiler";
    pred_tranform_func_ = PredTransformFunction("native", model_ptr);
    files_.clear();

    const char* predict_function_signature
      = (num_class_ > 1) ?
           "size_t predict_multiclass(union Entry* data, int pred_margin, float* result)"
         : "float predict(union Entry* data, int pred_margin)";

    std::ostringstream main_program;
    std::string accumulator_definition
      = (num_class_ > 1
         ? fmt::format("float sum[{num_class}] = {{0.0f}}",
             "num_class"_a = num_class_)
         : std::string("float sum = 0.0f"));

    std::string output_statement
      = (num_class_ > 1
         ? fmt::format("sum[tree_id % {num_class}] += tree[nid].info.leaf_value;",
             "num_class"_a = num_class_)
         : std::string("sum += tree[nid].info.leaf_value;"));

    std::string return_statement
      = (num_class_ > 1
         ? fmt::format(return_multiclass_template,
             "num_class"_a = num_class_,
             "global_bias"_a
                = compiler::common_util::ToStringHighPrecision(model.param.global_bias))
         : fmt::format(return_template,
             "global_bias"_a
                = compiler::common_util::ToStringHighPrecision(model.param.global_bias)));

    std::string nodes, nodes_row_ptr;
    std::vector<char> nodes_elf;
    if (param_.dump_array_as_elf > 0) {
      if (param_.verbose > 0) {
        TREELITE_LOG(INFO) << "Dumping arrays as an ELF relocatable object...";
      }
      std::tie(nodes_elf, nodes_row_ptr) = FormatNodesArrayELF(model);
    } else {
      std::tie(nodes, nodes_row_ptr) = FormatNodesArray(model);
    }

    const ModelParam model_param = model.param;
    const std::string query_functions_definition
      = fmt::format(native::query_functions_definition_template,
          "num_class"_a = num_class_,
          "num_feature"_a = num_feature_,
          "pred_transform"_a = model_param.pred_transform,
          "sigmoid_alpha"_a = model_param.sigmoid_alpha,
          "ratio_c"_a = model_param.ratio_c,
          "global_bias"_a = model_param.global_bias,
          "threshold_type_str"_a = TypeInfoToString(TypeToInfo<float>()),
          "leaf_output_type_str"_a = TypeInfoToString(TypeToInfo<float>()));

    main_program << fmt::format(main_template,
      "nodes_row_ptr"_a = nodes_row_ptr,
      "query_functions_definition"_a = query_functions_definition,
      "pred_transform_function"_a = pred_tranform_func_,
      "predict_function_signature"_a = predict_function_signature,
      "num_tree"_a = model.trees.size(),
      "compare_op"_a = GetCommonOp(model),
      "accumulator_definition"_a = accumulator_definition,
      "output_statement"_a = output_statement,
      "return_statement"_a = return_statement);

    files_["main.c"] = CompiledModel::FileEntry(main_program.str());

    if (param_.dump_array_as_elf > 0) {
      files_["arrays.o"] = CompiledModel::FileEntry(std::move(nodes_elf));
    } else {
      files_["arrays.c"] = CompiledModel::FileEntry(fmt::format(arrays_template,
        "nodes"_a = nodes));
    }

    const std::string query_functions_prototype
        = fmt::format(native::query_functions_prototype_template,
                      "dllexport"_a = DLLEXPORT_KEYWORD);
    files_["header.h"] = CompiledModel::FileEntry(fmt::format(header_template,
      "dllexport"_a = DLLEXPORT_KEYWORD,
      "query_functions_prototype"_a = query_functions_prototype,
      "predict_function_signature"_a = predict_function_signature));

    {
      /* write recipe.json */
      rapidjson::StringBuffer os;
      rapidjson::Writer<rapidjson::StringBuffer> writer(os);

      writer.StartObject();
      writer.Key("target");
      writer.String(param_.native_lib_name.data(), param_.native_lib_name.size());
      writer.Key("sources");
      writer.StartArray();
      std::vector<std::string> extra_file_list;
      for (const auto& kv : files_) {
        if (EndsWith(kv.first, ".c")) {
          const size_t line_count
            = std::count(kv.second.content.begin(), kv.second.content.end(), '\n');
          writer.StartObject();
          writer.Key("name");
          std::string name = kv.first.substr(0, kv.first.length() - 2);
          writer.String(name.data(), name.size());
          writer.Key("length");
          writer.Uint64(line_count);
          writer.EndObject();
        } else if (EndsWith(kv.first, ".o")) {
          extra_file_list.push_back(kv.first);
        }
      }
      writer.EndArray();
      if (!extra_file_list.empty()) {
        writer.Key("extra");
        writer.StartArray();
        for (const auto& extra_file : extra_file_list) {
          writer.String(extra_file.data(), extra_file.size());
        }
        writer.EndArray();
      }
      writer.EndObject();

      files_["recipe.json"] = CompiledModel::FileEntry(os.GetString());
    }
    cm.files = std::move(files_);
    return cm;
  }

  CompilerParam QueryParam() const {
    return param_;
  }

 private:
  CompilerParam param_;
  int num_feature_;
  unsigned int num_class_;
  std::string pred_tranform_func_;
  std::unordered_map<std::string, CompiledModel::FileEntry> files_;
};

FailSafeCompiler::FailSafeCompiler(const CompilerParam& param)
    : pimpl_(std::make_unique<FailSafeCompilerImpl>(param)) {
  if (param.verbose > 0) {
    TREELITE_LOG(INFO) << "Using FailSafeCompiler";
  }
  if (param.annotate_in != "NULL") {
    TREELITE_LOG(INFO) << "Warning: 'annotate_in' parameter is not applicable for "
                 "FailSafeCompiler";
  }
  if (param.quantize > 0) {
    TREELITE_LOG(INFO) << "Warning: 'quantize' parameter is not applicable for "
                 "FailSafeCompiler";
  }
  if (param.parallel_comp > 0) {
    TREELITE_LOG(INFO) << "Warning: 'parallel_comp' parameter is not applicable for "
                 "FailSafeCompiler";
  }
  if (std::isfinite(param.code_folding_req)) {
    TREELITE_LOG(INFO) << "Warning: 'code_folding_req' parameter is not applicable "
                 "for FailSafeCompiler";
  }
}

FailSafeCompiler::~FailSafeCompiler() = default;

CompiledModel
FailSafeCompiler::Compile(const Model& model) {
  return pimpl_->Compile(model);
}

CompilerParam
FailSafeCompiler::QueryParam() const {
  return pimpl_->QueryParam();
}

}  // namespace compiler
}  // namespace treelite
