#include <treelite/compiler.h>
#include <unordered_map>
#include "./param.h"
#include "./pred_transform.h"
#include "./ast/builder.h"
#include "./native/get_num_feature.h"
#include "./native/get_num_output_group.h"

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

    num_output_group_ = model.num_output_group;
    pred_tranform_func_ = PredTransformFunction("native", model);
    files_.clear();

    ASTBuilder builder;
    builder.Build(model);
    builder.Split(param.parallel_comp);
    if (param.quantize > 0) {
      builder.QuantizeThresholds();
    }
    files_["header.h"] = 
         "#include <stdlib.h>\n#include <string.h>\n"
         "#include <math.h>\n#include <stdint.h>\n\n"
         "union Entry {\n  int missing;\n  float fvalue;\n  int qvalue;\n};\n\n";
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
  int num_output_group_;
  std::string pred_tranform_func_;
  std::unordered_map<std::string, std::string> files_;

  void WalkAST(const ASTNode* node,
               const std::string& dest,
               int indent) {
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
      LOG(FATAL) << "oops";
    }
  }

  void HandleMainNode(const MainNode* node,
                      const std::string& dest,
                      int indent) {
    const std::string prototype
      = (num_output_group_ > 1) ?
          "size_t predict_multiclass(union Entry* data, int pred_margin, "
                                    "float* result)"
        : "float predict(union Entry* data, int pred_margin)";
    files_[dest] += std::string(indent, ' ') + "#include \"header.h\"\n\n";
    files_[dest] += get_num_output_group_func(num_output_group_) + "\n"
                    + get_num_feature_func(node->num_feature) + "\n"
                    + pred_tranform_func_ + "\n"
                    + std::string(indent, ' ') + prototype + " {\n";
    files_["header.h"] += get_num_output_group_func_prototype();
    files_["header.h"] += get_num_feature_func_prototype();
    files_["header.h"] += prototype + ";\n";
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent + 2);
    std::ostringstream oss;
    if (num_output_group_ > 1) {
      oss << std::string(indent + 2, ' ')
          << "for (int i = 0; i < " << num_output_group_ << "; ++i) {\n"
          << std::string(indent + 4, ' ') << "result[i] = sum[i]";
      if (node->average_result) {
        oss << " / " << node->num_tree;
      }
      oss << " + (float)(" << common::ToString(node->global_bias) << ");\n"
          << std::string(indent + 2, ' ') << "}\n"
          << std::string(indent + 2, ' ') << "if (!pred_margin) {\n"
          << std::string(indent + 2, ' ')
          << "  return pred_transform(result);\n"
          << std::string(indent + 2, ' ') << "} else {\n"
          << std::string(indent + 2, ' ')
          << "  return " << num_output_group_ << ";\n"
          << std::string(indent + 2, ' ') << "}\n";
    } else {
      oss << std::string(indent + 2, ' ') << "sum = sum";
      if (node->average_result) {
        oss << " / " << node->num_tree;
      }
      oss << " + (float)(" << common::ToString(node->global_bias) << ");\n"
          << std::string(indent + 2, ' ') << "if (!pred_margin) {\n"
          << std::string(indent + 2, ' ') << "  return pred_transform(sum);\n"
          << std::string(indent + 2, ' ') << "} else {\n"
          << std::string(indent + 2, ' ') << "  return sum;\n"
          << std::string(indent + 2, ' ') << "}\n";
    }
    oss << std::string(indent, ' ') << "}\n";
    files_[dest] += oss.str();
  }

  void HandleACNode(const AccumulatorContextNode* node,
                    const std::string& dest,
                    int indent) {
    std::ostringstream oss;
    if (num_output_group_ > 1) {
      oss << std::string(indent, ' ')
          << "float sum[" << num_output_group_ << "] = {0.0f};\n";
    } else {
      oss << std::string(indent, ' ') << "float sum = 0.0f;\n";
    }
    oss << std::string(indent, ' ') << "unsigned int tmp;\n";
    files_[dest] += oss.str();
    for (ASTNode* child : node->children) {
      WalkAST(child, dest, indent);
    }
  }

  void HandleCondNode(const ConditionNode* node,
                      const std::string& dest,
                      int indent) {
    const unsigned split_index = node->split_index;
    const std::string na_check
      = std::string("data[") + std::to_string(split_index) + "].missing != -1";
    
    const NumericalConditionNode* t;
    std::ostringstream oss;  // prepare logical statement for evaluating split
    if ( (t = dynamic_cast<const NumericalConditionNode*>(node)) ) {
      if (t->quantized) {
        oss << "data[" << split_index << "].qvalue "
            << OpName(t->op) << " " << t->threshold.int_val;
      } else {
        // to restore default precision
        const std::streamsize ss = std::cout.precision();
        oss << "data[" << split_index << "].fvalue "
            << OpName(t->op) << " "
            << std::setprecision(std::numeric_limits<tl_float>::digits10 + 2)
            << t->threshold.float_val << std::setprecision(ss);
      }
    } else {  // categorical split
      const CategoricalConditionNode* t2
        = dynamic_cast<const CategoricalConditionNode*>(node);
      CHECK(t2);
      std::vector<uint64_t> bitmap = to_bitmap(t2->left_categories);
      CHECK_GE(bitmap.size(), 1);
      bool all_zeros = true;
      for (uint64_t e : bitmap) {
        all_zeros &= (e == 0);
      }
      if (all_zeros) {
        oss << "0";
      } else {
        oss << "(tmp = (unsigned int)(data[" << split_index << "].fvalue) ), "
            << "(tmp >= 0 && tmp < 64 && (( (uint64_t)"
            << bitmap[0] << "U >> tmp) & 1) )";
        for (size_t i = 1; i < bitmap.size(); ++i) {
          oss << " || (tmp >= " << (i * 64)
              << " && tmp < " << ((i + 1) * 64)
              << " && (( (uint64_t)" << bitmap[i]
              << "U >> (tmp - " << (i * 64) << ") ) & 1) )";
        }
      }
    }
    files_[dest]
      += std::string(indent, ' ') + "if ("
         + ((node->default_left) ? (std::string("!(") + na_check + ") || (")
                                 : (std::string(" (") + na_check + ") && ("))
         + oss.str() + ") ) {\n";
    CHECK_EQ(node->children.size(), 2);
    WalkAST(node->children[0], dest, indent + 2);
    files_[dest] += std::string(indent, ' ') + "} else {\n";
    WalkAST(node->children[1], dest, indent + 2);
    files_[dest] += std::string(indent, ' ') + "}\n";
  }

  void HandleOutputNode(const OutputNode* node,
                        const std::string& dest,
                        int indent) {
    std::ostringstream oss;
    if (num_output_group_ > 1) {
      if (node->is_vector) {
        // multi-class classification with random forest
        const std::vector<tl_float>& leaf_vector = node->vector;
        CHECK_EQ(leaf_vector.size(), static_cast<size_t>(num_output_group_))
          << "Ill-formed model: leaf vector must be of length [num_output_group]";
        for (int group_id = 0; group_id < num_output_group_; ++group_id) {
          oss << std::string(indent, ' ')
              << "sum[" << group_id << "] += (float)"
              << common::ToString(leaf_vector[group_id]) << ";\n";
        }
      } else {
        // multi-class classification with gradient boosted trees
        oss << std::string(indent, ' ') << "sum["
            << (node->tree_id % num_output_group_) << "] += (float)"
            << common::ToString(node->scalar) << ";\n";
      }
    } else {
      oss << std::string(indent, ' ') << "sum += (float)"
          << common::ToString(node->scalar) << ";\n";
    }
    files_[dest] += oss.str();
    CHECK_EQ(node->children.size(), 0);
  }

  void HandleTUNode(const TranslationUnitNode* node,
                    const std::string& dest,
                    int indent) {
    const std::string new_file
      = std::string("tu") + std::to_string(node->unit_id) + ".c";
    std::ostringstream caller_buf, callee_buf, func_name, prototype;
    callee_buf << "#include \"header.h\"\n";
    if (num_output_group_ > 1) {
      func_name << "predict_margin_multiclass_unit" << node->unit_id;
      caller_buf << std::string(indent, ' ')
                 << func_name.str() << "(data, sum);\n";
      prototype << "void " << func_name.str()
                << "(union Entry* data, float* result)";
    } else {
      func_name << "predict_margin_unit" << node->unit_id;
      caller_buf << std::string(indent, ' ')
                 << "sum += " << func_name.str() << "(data);\n";
      prototype << "float " << func_name.str() << "(union Entry* data)";
    }
    callee_buf << prototype.str() << " {\n";
    files_[dest] += caller_buf.str();
    files_[new_file] += callee_buf.str();
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], new_file, 2);
    callee_buf.str(""); callee_buf.clear();
    if (num_output_group_ > 1) {
      callee_buf << "  for (int i = 0; i < " << num_output_group_ << "; ++i) {\n"
                 << "    result[i] = sum[i];\n"
                 << "  }\n";
    } else {
      callee_buf << "  return sum;\n";
    }
    callee_buf << "}\n";
    files_[new_file] += callee_buf.str();
    files_["header.h"] += prototype.str() + ";\n";
  }

  void HandleQNode(const QuantizerNode* node,
                   const std::string& dest,
                   int indent) {
    std::ostringstream oss;  // prepare a preamble
    const int num_feature = node->is_categorical.size();
    size_t length = 2;
    oss << "static const unsigned char is_categorical[] = {\n  ";
    for (int fid = 0; fid < num_feature; ++fid) {
      if (node->is_categorical[fid]) {
        common::WrapText(&oss, &length, "1, ", 2, 80);
      } else {
        common::WrapText(&oss, &length, "0, ", 2, 80);
      }
    }
    oss << "\n};\n";
    length = 2;
    oss << "static const float threshold[] = {\n  ";
    for (const auto& e : node->cut_pts) {
      for (tl_float v : e) {
        common::WrapText(&oss, &length, common::ToString(v) + ", ", 2, 80);
      }
    }
    oss << "\n};\n";
    length = 2;
    size_t accum = 0;
    oss << "static const int th_begin[] = {\n  ";
    for (const auto& e : node->cut_pts) {
      common::WrapText(&oss, &length, std::to_string(accum) + ", ", 2, 80);
      accum += e.size();
    }
    oss << "\n};\n";
    length = 2;
    oss << "static const int th_len[] = {\n  ";
    for (const auto& e : node->cut_pts) {
      common::WrapText(&oss, &length, std::to_string(e.size()) + ", ", 2, 80);
    }
    oss << "\n};\n";
    #include "./native/quantize_func.h"
    oss << quantize_func << files_[dest] << std::string(indent, ' ')
        << "for (int i = 0; i < " << num_feature << "; ++i) {\n"
        << std::string(indent + 2, ' ')
        << "if (data[i].missing != -1 && !is_categorical[i]) {\n"
        << std::string(indent + 4, ' ')
        << "data[i].qvalue = quantize(data[i].fvalue, i);\n"
        << std::string(indent + 2, ' ') + "}\n"
        << std::string(indent, ' ') + "}\n";
    files_[dest] = oss.str();
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent);
  }

  inline std::vector<uint64_t>
  to_bitmap(const std::vector<uint32_t>& left_categories) const {
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

TREELITE_REGISTER_COMPILER(ASTNativeCompiler, "ast_native")
.describe("AST-based compiler that produces C code")
.set_body([](const CompilerParam& param) -> Compiler* {
    return new ASTNativeCompiler(param);
  });
}  // namespace compiler
}  // namespace treelite
