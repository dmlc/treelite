/*!
 * Copyright (c) 2017 by Contributors
 * \file ast_native.cc
 * \author Philip Cho
 * \brief C code generator
 */
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <treelite/annotator.h>
#include <fmt/format.h>
#include "./pred_transform.h"
#include "./ast/builder.h"
#include "./native/main_template.h"
#include "./native/header_template.h"
#include "./native/qnode_template.h"
#include "./native/code_folder_template.h"
#include "./common/format_util.h"
#include "./common/code_folding_util.h"
#include "./common/categorical_bitmap.h"

#if defined(_MSC_VER) || defined(_WIN32)
#define DLLEXPORT_KEYWORD "__declspec(dllexport) "
#else
#define DLLEXPORT_KEYWORD ""
#endif

using namespace fmt::literals;

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(ast_native);

class ASTNativeCompiler : public Compiler {
 public:
  explicit ASTNativeCompiler(const CompilerParam& param)
    : param(param) {
    if (param.verbose > 0) {
      LOG(INFO) << "Using ASTNativeCompiler";
    }
    if (param.dump_array_as_elf > 0) {
      LOG(INFO) << "Warning: 'dump_array_as_elf' parameter is not applicable "
                   "for ASTNativeCompiler";
    }
  }

  CompiledModel Compile(const Model& model) override {
    CompiledModel cm;
    cm.backend = "native";

    num_feature_ = model.num_feature;
    num_output_group_ = model.num_output_group;
    pred_transform_ = model.param.pred_transform;
    sigmoid_alpha_ = model.param.sigmoid_alpha;
    global_bias_ = model.param.global_bias;
    pred_tranform_func_ = PredTransformFunction("native", model);
    files_.clear();

    ASTBuilder builder;
    builder.BuildAST(model);
    if (builder.FoldCode(param.code_folding_req)
        || param.quantize > 0) {
      // is_categorical[i] : is i-th feature categorical?
      array_is_categorical_
        = RenderIsCategoricalArray(builder.GenerateIsCategoricalArray());
    }
    if (param.annotate_in != "NULL") {
      BranchAnnotator annotator;
      std::unique_ptr<dmlc::Stream> fi(
        dmlc::Stream::Create(param.annotate_in.c_str(), "r"));
      annotator.Load(fi.get());
      const auto annotation = annotator.Get();
      builder.LoadDataCounts(annotation);
      LOG(INFO) << "Loading node frequencies from `"
                << param.annotate_in << "'";
    }
    builder.Split(param.parallel_comp);
    if (param.quantize > 0) {
      builder.QuantizeThresholds();
    }

    {
      const char* destfile = getenv("TREELITE_DUMP_AST");
      if (destfile) {
        std::ofstream os(destfile);
        os << builder.GetDump() << std::endl;
      }
    }

    WalkAST(builder.GetRootNode(), "main.c", 0);
    if (files_.count("arrays.c") > 0) {
      PrependToBuffer("arrays.c", "#include \"header.h\"\n", 0);
    }

    {
      /* write recipe.json */
      std::vector<std::unordered_map<std::string, std::string>> source_list;
      for (const auto& kv : files_) {
        if (kv.first.compare(kv.first.length() - 2, 2, ".c") == 0) {
          const size_t line_count
            = std::count(kv.second.content.begin(), kv.second.content.end(), '\n');
          source_list.push_back({ {"name",
                                   kv.first.substr(0, kv.first.length() - 2)},
                                  {"length", std::to_string(line_count)} });
        }
      }
      std::ostringstream oss;
      std::unique_ptr<dmlc::JSONWriter> writer(new dmlc::JSONWriter(&oss));
      writer->BeginObject();
      writer->WriteObjectKeyValue("target", param.native_lib_name);
      writer->WriteObjectKeyValue("sources", source_list);
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
  std::string pred_transform_;
  float sigmoid_alpha_;
  float global_bias_;
  std::string pred_tranform_func_;
  std::string array_is_categorical_;
  std::unordered_map<std::string, CompiledModel::FileEntry> files_;

  void WalkAST(const ASTNode* node,
               const std::string& dest,
               size_t indent) {
    const MainNode* t1;
    const AccumulatorContextNode* t2;
    const ConditionNode* t3;
    const OutputNode* t4;
    const TranslationUnitNode* t5;
    const QuantizerNode* t6;
    const CodeFolderNode* t7;
    if ( (t1 = dynamic_cast<const MainNode*>(node)) ) {
      HandleMainNode(t1, dest, indent);
    } else if ( (t2 = dynamic_cast<const AccumulatorContextNode*>(node)) ) {
      HandleACNode(t2, dest, indent);
    } else if ( (t3 = dynamic_cast<const ConditionNode*>(node)) ) {
      HandleCondNode(t3, dest, indent);
    } else if ( (t4 = dynamic_cast<const OutputNode*>(node)) ) {
      HandleOutputNode(t4, dest, indent);
    } else if ( (t5 = dynamic_cast<const TranslationUnitNode*>(node)) ) {
      HandleTUNode(t5, dest, indent);
    } else if ( (t6 = dynamic_cast<const QuantizerNode*>(node)) ) {
      HandleQNode(t6, dest, indent);
    } else if ( (t7 = dynamic_cast<const CodeFolderNode*>(node)) ) {
      HandleCodeFolderNode(t7, dest, indent);
    } else {
      LOG(FATAL) << "Unrecognized AST node type";
    }
  }

  // append content to a given buffer, with given level of indentation
  inline void AppendToBuffer(const std::string& dest,
                             const std::string& content,
                             size_t indent) {
    files_[dest].content += common_util::IndentMultiLineString(content, indent);
  }

  // prepend content to a given buffer, with given level of indentation
  inline void PrependToBuffer(const std::string& dest,
                              const std::string& content,
                              size_t indent) {
    files_[dest].content
      = common_util::IndentMultiLineString(content, indent) + files_[dest].content;
  }

  void HandleMainNode(const MainNode* node,
                      const std::string& dest,
                      size_t indent) {
    const char* get_num_output_group_function_signature
      = "size_t get_num_output_group(void)";
    const char* get_num_feature_function_signature
      = "size_t get_num_feature(void)";
    const char* get_pred_transform_function_signature
      = "const char* get_pred_transform(void)";
    const char* get_sigmoid_alpha_function_signature
      = "float get_sigmoid_alpha(void)";
    const char* get_global_bias_function_signature
      = "float get_global_bias(void)";
    const char* predict_function_signature
      = (num_output_group_ > 1) ?
          "size_t predict_multiclass(union Entry* data, int pred_margin, "
                                    "float* result)"
        : "float predict(union Entry* data, int pred_margin)";

    if (!array_is_categorical_.empty()) {
      array_is_categorical_
        = fmt::format("const unsigned char is_categorical[] = {{\n{}\n}}",
                      array_is_categorical_);
    }

    AppendToBuffer(dest,
      fmt::format(native::main_start_template,
        "array_is_categorical"_a = array_is_categorical_,
        "get_num_output_group_function_signature"_a
          = get_num_output_group_function_signature,
        "get_num_feature_function_signature"_a
          = get_num_feature_function_signature,
        "get_pred_transform_function_signature"_a
          = get_pred_transform_function_signature,
        "get_sigmoid_alpha_function_signature"_a
          = get_sigmoid_alpha_function_signature,
        "get_global_bias_function_signature"_a
          = get_global_bias_function_signature,
        "pred_transform_function"_a = pred_tranform_func_,
        "predict_function_signature"_a = predict_function_signature,
        "num_output_group"_a = num_output_group_,
        "num_feature"_a = num_feature_,
        "pred_transform"_a = pred_transform_,
        "sigmoid_alpha"_a = sigmoid_alpha_,
        "global_bias"_a = global_bias_),
      indent);
    AppendToBuffer("header.h",
      fmt::format(native::header_template,
        "dllexport"_a = DLLEXPORT_KEYWORD,
        "get_num_output_group_function_signature"_a
          = get_num_output_group_function_signature,
        "get_num_feature_function_signature"_a
          = get_num_feature_function_signature,
        "get_pred_transform_function_signature"_a
          = get_pred_transform_function_signature,
        "get_sigmoid_alpha_function_signature"_a
          = get_sigmoid_alpha_function_signature,
        "get_global_bias_function_signature"_a
          = get_global_bias_function_signature,
        "predict_function_signature"_a = predict_function_signature,
        "threshold_type"_a = (param.quantize > 0 ? "int" : "double")),
      indent);

    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent + 2);

    const std::string optional_average_field
      = (node->average_result) ? fmt::format(" / {}", node->num_tree)
                               : std::string("");
    if (num_output_group_ > 1) {
      AppendToBuffer(dest,
        fmt::format(native::main_end_multiclass_template,
          "num_output_group"_a = num_output_group_,
          "optional_average_field"_a = optional_average_field,
          "global_bias"_a = common_util::ToStringHighPrecision(node->global_bias)),
        indent);
    } else {
      AppendToBuffer(dest,
        fmt::format(native::main_end_template,
          "optional_average_field"_a = optional_average_field,
          "global_bias"_a = common_util::ToStringHighPrecision(node->global_bias)),
        indent);
    }
  }

  void HandleACNode(const AccumulatorContextNode* node,
                    const std::string& dest,
                    size_t indent) {
    if (num_output_group_ > 1) {
      AppendToBuffer(dest,
        fmt::format("float sum[{num_output_group}] = {{0.0f}};\n"
                    "unsigned int tmp;\n"
                    "int nid, cond, fid;  /* used for folded subtrees */\n",
          "num_output_group"_a = num_output_group_), indent);
    } else {
      AppendToBuffer(dest,
        "float sum = 0.0f;\n"
        "unsigned int tmp;\n"
        "int nid, cond, fid;  /* used for folded subtrees */\n", indent);
    }
    for (ASTNode* child : node->children) {
      WalkAST(child, dest, indent);
    }
  }

  void HandleCondNode(const ConditionNode* node,
                      const std::string& dest,
                      size_t indent) {
    const NumericalConditionNode* t;
    std::string condition, condition_with_na_check;
    if ( (t = dynamic_cast<const NumericalConditionNode*>(node)) ) {
      /* numerical split */
      condition = ExtractNumericalCondition(t);
      const char* condition_with_na_check_template
        = (node->default_left) ?
            "!(data[{split_index}].missing != -1) || ({condition})"
          : " (data[{split_index}].missing != -1) && ({condition})";
      condition_with_na_check
        = fmt::format(condition_with_na_check_template,
            "split_index"_a = node->split_index,
            "condition"_a = condition);
    } else {   /* categorical split */
      const CategoricalConditionNode* t2
        = dynamic_cast<const CategoricalConditionNode*>(node);
      CHECK(t2);
      condition_with_na_check = ExtractCategoricalCondition(t2);
    }
    if (node->children[0]->data_count && node->children[1]->data_count) {
      const int left_freq = node->children[0]->data_count.value();
      const int right_freq = node->children[1]->data_count.value();
      condition_with_na_check
        = fmt::format(" {keyword}( {condition} ) ",
            "keyword"_a = ((left_freq > right_freq) ? "LIKELY" : "UNLIKELY"),
            "condition"_a = condition_with_na_check);
    }
    AppendToBuffer(dest,
      fmt::format("if ({}) {{\n", condition_with_na_check), indent);
    CHECK_EQ(node->children.size(), 2);
    WalkAST(node->children[0], dest, indent + 2);
    AppendToBuffer(dest, "} else {\n", indent);
    WalkAST(node->children[1], dest, indent + 2);
    AppendToBuffer(dest, "}\n", indent);
  }

  void HandleOutputNode(const OutputNode* node,
                        const std::string& dest,
                        size_t indent) {
    AppendToBuffer(dest, RenderOutputStatement(node), indent);
    CHECK_EQ(node->children.size(), 0);
  }

  void HandleTUNode(const TranslationUnitNode* node,
                    const std::string& dest,
                    int indent) {
    const int unit_id = node->unit_id;
    const std::string new_file = fmt::format("tu{}.c", unit_id);

    std::string unit_function_name, unit_function_signature,
                unit_function_call_signature;
    if (num_output_group_ > 1) {
      unit_function_name
        = fmt::format("predict_margin_multiclass_unit{}", unit_id);
      unit_function_signature
        = fmt::format("void {}(union Entry* data, float* result)",
            unit_function_name);
      unit_function_call_signature
        = fmt::format("{}(data, sum);\n", unit_function_name);
    } else {
      unit_function_name
        = fmt::format("predict_margin_unit{}", unit_id);
      unit_function_signature
        = fmt::format("float {}(union Entry* data)", unit_function_name);
      unit_function_call_signature
        = fmt::format("sum += {}(data);\n", unit_function_name);
    }
    AppendToBuffer(dest, unit_function_call_signature, indent);
    AppendToBuffer(new_file,
                   fmt::format("#include \"header.h\"\n"
                               "{} {{\n", unit_function_signature), 0);
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], new_file, 2);
    if (num_output_group_ > 1) {
      AppendToBuffer(new_file,
        fmt::format("  for (int i = 0; i < {num_output_group}; ++i) {{\n"
                    "    result[i] += sum[i];\n"
                    "  }}\n"
                    "}}\n",
          "num_output_group"_a = num_output_group_), 0);
    } else {
      AppendToBuffer(new_file, "  return sum;\n}\n", 0);
    }
    AppendToBuffer("header.h", fmt::format("{};\n", unit_function_signature), 0);
  }

  void HandleQNode(const QuantizerNode* node,
                   const std::string& dest,
                   size_t indent) {
    /* render arrays needed to convert feature values into bin indices */
    std::string array_threshold, array_th_begin, array_th_len;
    // threshold[] : list of all thresholds that occur at least once in the
    //   ensemble model. For each feature, an ascending list of unique
    //   thresholds is generated. The range th_begin[i]:(th_begin[i]+th_len[i])
    //   of the threshold[] array stores the threshold list for feature i.
    size_t total_num_threshold;
      // to hold total number of (distinct) thresholds
    {
      common_util::ArrayFormatter formatter(80, 2);
      for (const auto& e : node->cut_pts) {
        // cut_pts had been generated in ASTBuilder::QuantizeThresholds
        // cut_pts[i][k] stores the k-th threshold of feature i.
        for (tl_float v : e) {
          formatter << v;
        }
      }
      array_threshold = formatter.str();
    }
    {
      common_util::ArrayFormatter formatter(80, 2);
      size_t accum = 0;  // used to compute cumulative sum over threshold counts
      for (const auto& e : node->cut_pts) {
        formatter << accum;
        accum += e.size();  // e.size() = number of thresholds for each feature
      }
      total_num_threshold = accum;
      array_th_begin = formatter.str();
    }
    {
      common_util::ArrayFormatter formatter(80, 2);
      for (const auto& e : node->cut_pts) {
        formatter << e.size();
      }
      array_th_len = formatter.str();
    }
    if (!array_threshold.empty() && !array_th_begin.empty() && !array_th_len.empty()) {
      PrependToBuffer(dest,
        fmt::format(native::qnode_template,
          "total_num_threshold"_a = total_num_threshold), 0);
      AppendToBuffer(dest,
        fmt::format(native::quantize_loop_template,
          "num_feature"_a = num_feature_), indent);
    }
    if (!array_threshold.empty()) {
      PrependToBuffer(dest,
        fmt::format("static const double threshold[] = {{\n"
                    "{array_threshold}\n"
                    "}};\n", "array_threshold"_a = array_threshold), 0);
    }
    if (!array_th_begin.empty()) {
      PrependToBuffer(dest,
        fmt::format("static const int th_begin[] = {{\n"
                    "{array_th_begin}\n"
                    "}};\n", "array_th_begin"_a = array_th_begin), 0);
    }
    if (!array_th_len.empty()) {
      PrependToBuffer(dest,
        fmt::format("static const int th_len[] = {{\n"
                    "{array_th_len}\n"
                    "}};\n", "array_th_len"_a = array_th_len), 0);
    }
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent);
  }

  void HandleCodeFolderNode(const CodeFolderNode* node,
                            const std::string& dest,
                            size_t indent) {
    CHECK_EQ(node->children.size(), 1);
    const int node_id = node->children[0]->node_id;
    const int tree_id = node->children[0]->tree_id;

    /* render arrays needed for folding subtrees */
    std::string array_nodes, array_cat_bitmap, array_cat_begin;
    // node_treeXX_nodeXX[] : information of nodes for a particular subtree
    const std::string node_array_name
      = fmt::format("node_tree{}_node{}", tree_id, node_id);
    // cat_bitmap_treeXX_nodeXX[] : list of all 64-bit integer bitmaps, used to
    //                              make all categorical splits in a particular
    //                              subtree
    const std::string cat_bitmap_name
      = fmt::format("cat_bitmap_tree{}_node{}", tree_id, node_id);
    // cat_begin_treeXX_nodeXX[] : shows which bitmaps belong to each split.
    //                             cat_bitmap[ cat_begin[i]:cat_begin[i+1] ]
    //                             belongs to the i-th (categorical) split
    const std::string cat_begin_name
      = fmt::format("cat_begin_tree{}_node{}", tree_id, node_id);

    std::string output_switch_statement;
    Operator common_comp_op;
    common_util::RenderCodeFolderArrays(node, param.quantize, false,
      "{{ {default_left}, {split_index}, {threshold}, {left_child}, {right_child} }}",
      [this](const OutputNode* node) { return RenderOutputStatement(node); },
      &array_nodes, &array_cat_bitmap, &array_cat_begin,
      &output_switch_statement, &common_comp_op);
    if (!array_nodes.empty()) {
      AppendToBuffer("header.h",
                     fmt::format("extern const struct Node {node_array_name}[];\n",
                       "node_array_name"_a = node_array_name), 0);
      AppendToBuffer("arrays.c",
                     fmt::format("const struct Node {node_array_name}[] = {{\n"
                                 "{array_nodes}\n"
                                 "}};\n",
                       "node_array_name"_a = node_array_name,
                       "array_nodes"_a = array_nodes), 0);
    }

    if (!array_cat_bitmap.empty()) {
      AppendToBuffer("header.h",
                     fmt::format("extern const uint64_t {cat_bitmap_name}[];\n",
                       "cat_bitmap_name"_a = cat_bitmap_name), 0);
      AppendToBuffer("arrays.c",
                     fmt::format("const uint64_t {cat_bitmap_name}[] = {{\n"
                                 "{array_cat_bitmap}\n"
                                 "}};\n",
                       "cat_bitmap_name"_a = cat_bitmap_name,
                       "array_cat_bitmap"_a = array_cat_bitmap), 0);
    }

    if (!array_cat_begin.empty()) {
      AppendToBuffer("header.h",
                     fmt::format("extern const size_t {cat_begin_name}[];\n",
                       "cat_begin_name"_a = cat_begin_name), 0);
      AppendToBuffer("arrays.c",
                     fmt::format("const size_t {cat_begin_name}[] = {{\n"
                                 "{array_cat_begin}\n"
                                 "}};\n",
                       "cat_begin_name"_a = cat_begin_name,
                       "array_cat_begin"_a = array_cat_begin), 0);
    }

    if (array_nodes.empty()) {
      /* folded code consists of a single leaf node */
      AppendToBuffer(dest,
                     fmt::format("nid = -1;\n"
                                 "{output_switch_statement}\n",
                       "output_switch_statement"_a
                         = output_switch_statement), indent);
    } else if (!array_cat_bitmap.empty() && !array_cat_begin.empty()) {
      AppendToBuffer(dest,
                     fmt::format(native::eval_loop_template,
                       "node_array_name"_a = node_array_name,
                       "cat_bitmap_name"_a = cat_bitmap_name,
                       "cat_begin_name"_a = cat_begin_name,
                       "data_field"_a = (param.quantize > 0 ? "qvalue" : "fvalue"),
                       "comp_op"_a = OpName(common_comp_op),
                       "output_switch_statement"_a
                         = output_switch_statement), indent);
    } else {
      AppendToBuffer(dest,
                     fmt::format(native::eval_loop_template_without_categorical_feature,
                       "node_array_name"_a = node_array_name,
                       "data_field"_a = (param.quantize > 0 ? "qvalue" : "fvalue"),
                       "comp_op"_a = OpName(common_comp_op),
                       "output_switch_statement"_a
                         = output_switch_statement), indent);
    }
  }

  inline std::string
  ExtractNumericalCondition(const NumericalConditionNode* node) {
    std::string result;
    if (node->quantized) {  // quantized threshold
      result = fmt::format("data[{split_index}].qvalue {opname} {threshold}",
                 "split_index"_a = node->split_index,
                 "opname"_a = OpName(node->op),
                 "threshold"_a = node->threshold.int_val);
    } else if (std::isinf(node->threshold.float_val)) {  // infinite threshold
      // According to IEEE 754, the result of comparison [lhs] < infinity
      // must be identical for all finite [lhs]. Same goes for operator >.
      result = (CompareWithOp(0.0, node->op, node->threshold.float_val)
                ? "1" : "0");
    } else {  // finite threshold
      result = fmt::format("data[{split_index}].fvalue {opname} {threshold}",
                 "split_index"_a = node->split_index,
                 "opname"_a = OpName(node->op),
                 "threshold"_a = common_util::ToStringHighPrecision(node->threshold.float_val));
    }
    return result;
  }

  inline std::string
  ExtractCategoricalCondition(const CategoricalConditionNode* node) {
    std::string result;
    std::vector<uint64_t> bitmap
      = common_util::GetCategoricalBitmap(node->left_categories);
    CHECK_GE(bitmap.size(), 1);
    bool all_zeros = true;
    for (uint64_t e : bitmap) {
      all_zeros &= (e == 0);
    }
    if (all_zeros) {
      result = "0";
    } else {
      std::ostringstream oss;
      if (node->convert_missing_to_zero) {
        // All missing values are converted into zeros
        oss << fmt::format(
          "((tmp = (data[{0}].missing == -1 ? 0U "
          ": (unsigned int)(data[{0}].fvalue) )), ", node->split_index);
      } else {
        if (node->default_left) {
          oss << fmt::format(
            "data[{0}].missing == -1 || ("
            "(tmp = (unsigned int)(data[{0}].fvalue) ), ", node->split_index);
        } else {
          oss << fmt::format(
            "data[{0}].missing != -1 && ("
            "(tmp = (unsigned int)(data[{0}].fvalue) ), ", node->split_index);
        }
      }
      oss << "(tmp >= 0 && tmp < 64 && (( (uint64_t)"
          << bitmap[0] << "U >> tmp) & 1) )";
      for (size_t i = 1; i < bitmap.size(); ++i) {
        oss << " || (tmp >= " << (i * 64)
            << " && tmp < " << ((i + 1) * 64)
            << " && (( (uint64_t)" << bitmap[i]
            << "U >> (tmp - " << (i * 64) << ") ) & 1) )";
      }
      oss << ")";
      result = oss.str();
    }
    return result;
  }

  inline std::string
  RenderIsCategoricalArray(const std::vector<bool>& is_categorical) {
    common_util::ArrayFormatter formatter(80, 2);
    for (int fid = 0; fid < num_feature_; ++fid) {
      formatter << (is_categorical[fid] ? 1 : 0);
    }
    return formatter.str();
  }

  inline std::string RenderOutputStatement(const OutputNode* node) {
    std::string output_statement;
    if (num_output_group_ > 1) {
      if (node->is_vector) {
        // multi-class classification with random forest
        CHECK_EQ(node->vector.size(), static_cast<size_t>(num_output_group_))
          << "Ill-formed model: leaf vector must be of length [num_output_group]";
        for (int group_id = 0; group_id < num_output_group_; ++group_id) {
          output_statement
            += fmt::format("sum[{group_id}] += (float){output};\n",
                 "group_id"_a = group_id,
                 "output"_a = common_util::ToStringHighPrecision(node->vector[group_id]));
        }
      } else {
        // multi-class classification with gradient boosted trees
        output_statement
          = fmt::format("sum[{group_id}] += (float){output};\n",
              "group_id"_a = node->tree_id % num_output_group_,
              "output"_a = common_util::ToStringHighPrecision(node->scalar));
      }
    } else {
      output_statement
        = fmt::format("sum += (float){output};\n",
            "output"_a = common_util::ToStringHighPrecision(node->scalar));
    }
    return output_statement;
  }
};

TREELITE_REGISTER_COMPILER(ASTNativeCompiler, "ast_native")
.describe("AST-based compiler that produces C code")
.set_body([](const CompilerParam& param) -> Compiler* {
    return new ASTNativeCompiler(param);
  });
}  // namespace compiler
}  // namespace treelite
