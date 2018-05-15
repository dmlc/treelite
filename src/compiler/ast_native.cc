#include <treelite/compiler.h>
#include <treelite/common.h>
#include <treelite/annotator.h>
#include <fmt/format.h>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <cmath>
#include "./param.h"
#include "./pred_transform.h"
#include "./ast/builder.h"

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
  }

  CompiledModel Compile(const Model& model) override {
    CompiledModel cm;
    cm.backend = "native";
    cm.files["main.c"] = "";

    num_feature_ = model.num_feature;
    num_output_group_ = model.num_output_group;
    pred_tranform_func_ = PredTransformFunction("native", model);
    files_.clear();

    ASTBuilder builder;
    builder.BuildAST(model);
    is_categorical_ = builder.GenerateIsCategoricalArray();
    builder.FoldCode(param.code_folding_data_count_req,
                     param.code_folding_sum_hess_req);
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
    if (param.ast_dump_path != "NULL") {
      builder.Serialize(param.ast_dump_path, param.ast_dump_binary > 0);
    }
    WalkAST(builder.GetRootNode(), "main.c", 0);

    {
      /* write recipe.json */
      std::vector<std::unordered_map<std::string, std::string>> source_list;
      for (auto kv : files_) {
        if (kv.first.compare(kv.first.length() - 2, 2, ".c") == 0) {
          const size_t line_count
            = std::count(kv.second.begin(), kv.second.end(), '\n');
          source_list.push_back({ {"name",
                                   kv.first.substr(0, kv.first.length() - 2)},
                                  {"length", std::to_string(line_count)} });
        }
      }
      std::ostringstream oss;
      auto writer = common::make_unique<dmlc::JSONWriter>(&oss);
      writer->BeginObject();
      writer->WriteObjectKeyValue("target", std::string("predictor"));
      writer->WriteObjectKeyValue("sources", source_list);
      writer->EndObject();
      files_["recipe.json"] = oss.str();
    }
    cm.files = std::move(files_);
    return cm;
  }
 private:
  CompilerParam param;
  int num_feature_;
  int num_output_group_;
  std::vector<bool> is_categorical_;
  std::string pred_tranform_func_;
  std::unordered_map<std::string, std::string> files_;

  void WalkAST(const ASTNode* node,
               const std::string& dest,
               size_t indent) {
    const MainNode* t1;
    const AccumulatorContextNode* t2;
    const ConditionNode* t3;
    const OutputNode* t4;
    const TranslationUnitNode* t5;
    const QuantizerNode* t6;
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
    } else {
      LOG(FATAL) << "Unrecognized AST node type";
    }
  }

  // append content to a given buffer, with given level of indentation
  inline void AppendToBuffer(const std::string& dest,
                             const std::string& content,
                             size_t indent) {
    files_[dest] += common::IndentMultiLineString(content, indent);
  }

  // prepend content to a given buffer, with given level of indentation
  inline void PrependToBuffer(const std::string& dest,
                              const std::string& content,
                              size_t indent) {
    files_[dest]
      = common::IndentMultiLineString(content, indent) + files_[dest];
  }

  void HandleMainNode(const MainNode* node,
                      const std::string& dest,
                      size_t indent) {
    const char* get_num_output_group_function_signature
      = "size_t get_num_output_group(void)";
    const char* get_num_feature_function_signature
      = "size_t get_num_feature(void)";
    const char* predict_function_signature
      = (num_output_group_ > 1) ?
          "size_t predict_multiclass(union Entry* data, int pred_margin, "
                                    "float* result)"
        : "float predict(union Entry* data, int pred_margin)";

    #include "./native/main_template.h"
    #include "./native/header_template.h"
    AppendToBuffer(dest,
      fmt::format(main_start_template,
        "get_num_output_group_function_signature"_a
          = get_num_output_group_function_signature,
        "get_num_feature_function_signature"_a
          = get_num_feature_function_signature,
        "pred_transform_function"_a = pred_tranform_func_,
        "predict_function_signature"_a = predict_function_signature,
        "num_output_group"_a = num_output_group_,
        "num_feature"_a = node->num_feature),
      indent);
    AppendToBuffer("header.h",
      fmt::format(header_template,
        "get_num_output_group_function_signature"_a
          = get_num_output_group_function_signature,
        "get_num_feature_function_signature"_a
          = get_num_feature_function_signature,
        "predict_function_signature"_a = predict_function_signature),
      indent);

    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent + 2);

    const std::string optional_average_field
      = (node->average_result) ? fmt::format(" / {}", node->num_tree)
                               : std::string("");
    if (num_output_group_ > 1) {
      AppendToBuffer(dest,
        fmt::format(main_end_multiclass_template,
          "num_output_group"_a = num_output_group_,
          "optional_average_field"_a = optional_average_field,
          "global_bias"_a = common::ToStringHighPrecision(node->global_bias)),
        indent);
    } else {
      AppendToBuffer(dest,
        fmt::format(main_end_template,
          "optional_average_field"_a = optional_average_field,
          "global_bias"_a = common::ToStringHighPrecision(node->global_bias)),
        indent);
    }
  }

  void HandleACNode(const AccumulatorContextNode* node,
                    const std::string& dest,
                    size_t indent) {
    if (num_output_group_ > 1) {
      AppendToBuffer(dest,
        fmt::format("float sum[{num_output_group}] = {{0.0f}};\n"
                    "unsigned int tmp;\n",
          "num_output_group"_a = num_output_group_), indent);
    } else {
      AppendToBuffer(dest,
        "float sum = 0.0f;\n"
        "unsigned int tmp;\n", indent);
    }
    for (ASTNode* child : node->children) {
      WalkAST(child, dest, indent);
    }
  }

  void HandleCondNode(const ConditionNode* node,
                      const std::string& dest,
                      size_t indent) {
    const NumericalConditionNode* t;
    std::string condition;
    if ( (t = dynamic_cast<const NumericalConditionNode*>(node)) ) {
      /* numerical split */
      condition = ExtractNumericalCondition(t);
    } else {   /* categorical split */
      const CategoricalConditionNode* t2
        = dynamic_cast<const CategoricalConditionNode*>(node);
      CHECK(t2);
      condition = ExtractCategoricalCondition(t2);
    }
    const char* condition_with_na_check_template
      = (node->default_left) ?
          "!(data[{split_index}].missing != -1) || ({condition})"
        : " (data[{split_index}].missing != -1) && ({condition})";
    std::string condition_with_na_check
      = fmt::format(condition_with_na_check_template,
          "split_index"_a = node->split_index,
          "condition"_a = condition);
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
                 "output"_a
                   = common::ToStringHighPrecision(node->vector[group_id]));
        }
      } else {
        // multi-class classification with gradient boosted trees
        output_statement
          = fmt::format("sum[{group_id}] += (float){output};\n",
              "group_id"_a = node->tree_id % num_output_group_,
              "output"_a = common::ToStringHighPrecision(node->scalar));
      }
    } else {
      output_statement
        = fmt::format("sum += (float){output};\n",
            "output"_a = common::ToStringHighPrecision(node->scalar));
    }
    AppendToBuffer(dest, output_statement, indent);
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
                    "    result[i] = sum[i];\n"
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
    std::string array_is_categorical, array_threshold,
                array_th_begin, array_th_len;
    // is_categorical[i] : is i-th feature categorical?
    {
      common::ArrayFormatter formatter(80, 2);
      for (int fid = 0; fid < num_feature_; ++fid) {
        formatter << (is_categorical_[fid] ? 1 : 0);
      }
      array_is_categorical = formatter.str();
    }
    // threshold[] : list of all thresholds that occur at least once in the
    //   ensemble model. For each feature, an ascending list of unique
    //   thresholds is generated. The range th_begin[i]:(th_begin[i]+th_len[i])
    //   of the threshold[] array stores the threshold list for feature i.
    size_t total_num_threshold;
      // to hold total number of (distinct) thresholds
    {
      common::ArrayFormatter formatter(80, 2);
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
      common::ArrayFormatter formatter(80, 2);
      size_t accum = 0;  // used to compute cumulative sum over threshold counts
      for (const auto& e : node->cut_pts) {
        formatter << accum;
        accum += e.size();  // e.size() = number of thresholds for each feature
      }
      total_num_threshold = accum;
      array_th_begin = formatter.str();
    }
    {
      common::ArrayFormatter formatter(80, 2);
      for (const auto& e : node->cut_pts) {
        formatter << e.size();
      }
      array_th_len = formatter.str();
    }

    #include "./native/qnode_template.h"
    PrependToBuffer(dest,
      fmt::format(qnode_template,
        "array_is_categorical"_a = array_is_categorical,
        "array_threshold"_a = array_threshold,
        "array_th_begin"_a = array_th_begin,
        "array_th_len"_a = array_th_len,
        "total_num_threshold"_a = total_num_threshold), 0);
    AppendToBuffer(dest,
      fmt::format(quantize_loop_template,
        "num_feature"_a = num_feature_), indent);
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent);
  }

  inline std::vector<uint64_t>
  to_bitmap(const std::vector<uint32_t>& left_categories) const {
    const size_t num_left_categories = left_categories.size();
    if (num_left_categories == 0) {
      return std::vector<uint64_t>{0};
    }
    const uint32_t max_left_category = left_categories[num_left_categories - 1];
    std::vector<uint64_t> bitmap((max_left_category + 1 + 63) / 64, 0);
    for (size_t i = 0; i < num_left_categories; ++i) {
      const uint32_t cat = left_categories[i];
      const size_t idx = cat / 64;
      const uint32_t offset = cat % 64;
      bitmap[idx] |= (static_cast<uint64_t>(1) << offset);
    }
    return bitmap;
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
      result = (common::CompareWithOp(0.0, node->op, node->threshold.float_val)
                ? "1" : "0");
    } else {  // finite threshold
      result = fmt::format("data[{split_index}].fvalue {opname} {threshold}",
                 "split_index"_a = node->split_index,
                 "opname"_a = OpName(node->op),
                 "threshold"_a
                   = common::ToStringHighPrecision(node->threshold.float_val));
    }
    return result;
  }

  inline std::string
  ExtractCategoricalCondition(const CategoricalConditionNode* node) {
    std::string result;
    std::vector<uint64_t> bitmap = to_bitmap(node->left_categories);
    CHECK_GE(bitmap.size(), 1);
    bool all_zeros = true;
    for (uint64_t e : bitmap) {
      all_zeros &= (e == 0);
    }
    if (all_zeros) {
      result = "0";
    } else {
      std::ostringstream oss;
      oss << "(tmp = (unsigned int)(data[" << node->split_index << "].fvalue) ), "
          << "(tmp >= 0 && tmp < 64 && (( (uint64_t)"
          << bitmap[0] << "U >> tmp) & 1) )";
      for (size_t i = 1; i < bitmap.size(); ++i) {
        oss << " || (tmp >= " << (i * 64)
            << " && tmp < " << ((i + 1) * 64)
            << " && (( (uint64_t)" << bitmap[i]
            << "U >> (tmp - " << (i * 64) << ") ) & 1) )";
      }
      result = oss.str();
      return result;
    }
  }
};

TREELITE_REGISTER_COMPILER(ASTNativeCompiler, "ast_native")
.describe("AST-based compiler that produces C code")
.set_body([](const CompilerParam& param) -> Compiler* {
    return new ASTNativeCompiler(param);
  });
}  // namespace compiler
}  // namespace treelite
