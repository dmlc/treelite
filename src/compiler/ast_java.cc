#include <treelite/compiler.h>
#include <unordered_map>
#include "./param.h"
#include "./ast/builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(ast_java);

class ASTJavaCompiler : public Compiler {
 public:
  explicit ASTJavaCompiler(const CompilerParam& param)
    : param(param) {
    if (param.verbose > 0) {
      LOG(INFO) << "Using ASTJavaCompiler";
    }
  }

  CompiledModel Compile(const Model& model) override {
    CompiledModel cm;
    cm.backend = "java";
    cm.files["Main.java"] = "";

    num_output_group_ = model.num_output_group;
    files_.clear();

    ASTBuilder builder;
    builder.Build(model);
    builder.Split(param.parallel_comp);
    if (param.quantize > 0) {
      builder.QuantizeThresholds();
    }
    builder.CountDescendant();
    builder.BreakUpLargeUnits(param.max_unit_size);
    #include "java/entry_type.h"
    #include "java/pom_xml.h"
    files_[file_prefix_ + "Entry.java"] = entry_type;
    files_["pom.xml"] = pom_xml;
    WalkAST(builder.GetRootNode(), "Main.java", 0);

    cm.files = std::move(files_);
    cm.file_prefix = file_prefix_;
    return cm;
  }
 private:
  CompilerParam param;
  int num_output_group_;
  std::unordered_map<std::string, std::string> files_;
  std::string main_tail_;
  const std::string file_prefix_ = "src/main/java/treelite/predictor/";

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

  void CommitToFile(const std::string& dest,
                    const std::string& content) {
    files_[file_prefix_ + dest] += content;
  }

  void HandleMainNode(const MainNode* node,
                      const std::string& dest,
                      int indent) {
    const std::string prototype
      = (num_output_group_ > 1) ?
          "public static void predict_margin_multiclass(Entry[] data, float[] result)"
        : "public static float predict_margin(Entry[] data)";
    CommitToFile(dest,
                 "package treelite.predictor;\n\n"
                 "import javolution.context.LogContext;\n"
                 "import javolution.context.LogContext.Level;\n\n"
                 "public class Main {\n");
    CommitToFile(dest,
                 "  static {\n    LogContext ctx = LogContext.enter();\n"
                 "    ctx.setLevel(Level.INFO);\n  }\n");
    CommitToFile(dest,
                 std::string(indent + 2, ' ') + prototype + " {\n");
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], dest, indent + 4);
    std::ostringstream oss;
    if (num_output_group_ > 1) {
      oss << std::string(indent + 4, ' ')
          << "for (int i = 0; i < " << num_output_group_ + "; ++i) {\n"
          << std::string(indent + 6, ' ') << "result[i] = sum[i]";
      if (node->average_result) {
        oss << " / " << node->num_tree;
      }
      oss << " + (" + common::ToString(node->global_bias) + ");\n"
          << std::string(indent + 4, ' ') + "}\n";
    } else {
      oss << std::string(indent + 4, ' ') + "return sum";
      if (node->average_result) {
        oss << " / " << node->num_tree;
      }
      oss << " + (" + common::ToString(node->global_bias) + ");\n";
    }
    oss << std::string(indent + 2, ' ') << "}\n"
        << main_tail_
        << std::string(indent, ' ') << "}\n";
    CommitToFile(dest, oss.str());
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
    oss << std::string(indent, ' ') << "int tmp;\n";
    CommitToFile(dest, oss.str());

    for (ASTNode* child : node->children) {
      WalkAST(child, dest, indent);
    }
  }

  void HandleCondNode(const ConditionNode* node,
                      const std::string& dest,
                      int indent) {
    const unsigned split_index = node->split_index;
    const std::string na_check
      = std::string("data[") + std::to_string(split_index)
        + "].missing.get() != -1";
    
    const NumericalConditionNode* t;
    std::ostringstream oss;  // prepare logical statement for evaluating split
    if ( (t = dynamic_cast<const NumericalConditionNode*>(node)) ) {
      if (t->quantized) {
        oss << "data[" << split_index << "].qvalue.get() "
            << OpName(t->op) << " " << t->threshold.int_val;
      } else {
        // to restore default precision
        const std::streamsize ss = std::cout.precision();
        oss << "data[" << split_index << "].fvalue.get() "
            << OpName(t->op) << " "
            << std::setprecision(std::numeric_limits<tl_float>::digits10 + 2)
            << t->threshold.float_val << "f" << std::setprecision(ss);
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
        oss << "(tmp = (int)(data[" << split_index << "].fvalue.get()) ), "
            << "(tmp >= 0 && tmp < 64 && (( (long)"
            << bitmap[0] << "L >>> tmp) & 1) )";
        for (size_t i = 1; i < bitmap.size(); ++i) {
          oss << " || (tmp >= " << (i * 64)
              << " && tmp < " << ((i + 1) * 64)
              << " && (( (long)" << bitmap[i]
              << "L >>> (tmp - " << (i * 64) << ") ) & 1) )";
        }
      }
    }
    CommitToFile(dest, 
         std::string(indent, ' ') + "if ("
         + ((node->default_left) ? (std::string("!(") + na_check + ") || (")
                                 : (std::string(" (") + na_check + ") && ("))
         + oss.str() + ") ) {\n");
    CHECK_EQ(node->children.size(), 2);
    WalkAST(node->children[0], dest, indent + 2);
    CommitToFile(dest, std::string(indent, ' ') + "} else {\n");
    WalkAST(node->children[1], dest, indent + 2);
    CommitToFile(dest, std::string(indent, ' ') + "}\n");
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
              << "sum[" << group_id << "] += "
              << common::ToString(leaf_vector[group_id]) << "f;\n";
        }
      } else {
        // multi-class classification with gradient boosted trees
        oss << std::string(indent, ' ') << "sum["
            << (node->tree_id % num_output_group_) << "] += "
            << common::ToString(node->scalar) << "f;\n";
      }
    } else {
      oss << std::string(indent, ' ') << "sum += "
          << common::ToString(node->scalar) << "f;\n";
    }
    CommitToFile(dest, oss.str());
    CHECK_EQ(node->children.size(), 0);
  }

  void HandleTUNode(const TranslationUnitNode* node,
                    const std::string& dest,
                    int indent) {
    const std::string new_file
      = std::string("TU") + std::to_string(node->unit_id) + ".java";
    std::ostringstream caller_buf, callee_buf, func_name, class_name, prototype;
    class_name << "TU" << node->unit_id;
    if (num_output_group_ > 1) {
      func_name << "predict_margin_multiclass_unit" << node->unit_id;
      caller_buf << std::string(indent, ' ')
                 << class_name.str() << "."
                 << func_name.str() << "(data, sum);\n";
      prototype << "public static void " << func_name.str()
                << "(Entry[] data, float[] result)";
    } else {
      func_name << "predict_margin_unit" << node->unit_id;
      caller_buf << std::string(indent, ' ')
                 << "sum += " << class_name.str() << "." << func_name.str()
                 << "(data);\n";
      prototype << "public static float " << func_name.str()
                << "(Entry[] data)";
    }
    callee_buf << "package treelite.predictor;\n\n"
               << "public class " << class_name.str() << " {\n"
               << "  " << prototype.str() << " {\n";
    CommitToFile(dest, caller_buf.str());
    CommitToFile(new_file, callee_buf.str());
    CHECK_EQ(node->children.size(), 1);
    WalkAST(node->children[0], new_file, 4);
    callee_buf.str(""); callee_buf.clear();
    if (num_output_group_ > 1) {
      callee_buf
        << "    for (int i = 0; i < " << num_output_group_ << "; ++i) {\n"
        << "      result[i] = sum[i];\n"
        << "    }\n";
    } else {
      callee_buf << "    return sum;\n";
    }
    callee_buf << "  }\n" << "}\n";
    CommitToFile(new_file, callee_buf.str());
  }

  void HandleQNode(const QuantizerNode* node,
                   const std::string& dest,
                   int indent) {
    std::ostringstream oss;  // prepare a preamble
    const int num_feature = node->is_categorical.size();
    size_t length = 4;
    oss << "  private static final boolean[] is_categorical = {\n    ";
    for (int fid = 0; fid < num_feature; ++fid) {
      if (node->is_categorical[fid]) {
        common::WrapText(&oss, &length, "true, ", 4, 80);
      } else {
        common::WrapText(&oss, &length, "false, ", 4, 80);
      }
    }
    oss << "\n  };\n";
    length = 4;
    oss << "  private static final float[] threshold = {\n    ";
    for (const auto& e : node->cut_pts) {
      for (tl_float v : e) {
        common::WrapText(&oss, &length, common::ToString(v) + "f, ", 4, 80);
      }
    }
    oss << "\n  };\n";
    length = 4;
    size_t accum = 0;
    oss << "  private static final int[] th_begin = {\n    ";
    for (const auto& e : node->cut_pts) {
      common::WrapText(&oss, &length, std::to_string(accum) + ", ", 4, 80);
      accum += e.size();
    }
    oss << "\n  };\n";
    length = 4;
    oss << "  private static final int[] th_len = {\n    ";
    for (const auto& e : node->cut_pts) {
      common::WrapText(&oss, &length, std::to_string(e.size()) + ", ", 4, 80);
    }
    oss << "\n  };\n";
    #include "./java/quantize_func.h"
    oss << quantize_func;
    main_tail_ += oss.str();
    oss.str(""); oss.clear();
    oss << std::string(indent, ' ')
        << "for (int i = 0; i < " << num_feature << "; ++i) {\n"
        << std::string(indent + 2, ' ')
        << "if (data[i].missing.get() != -1 && !is_categorical[i]) {\n"
        << std::string(indent + 4, ' ')
        << "data[i].qvalue.set(quantize(data[i].fvalue.get(), i));\n"
        << std::string(indent + 2, ' ') + "}\n"
        << std::string(indent, ' ') + "}\n";
    CommitToFile(dest, oss.str());
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

TREELITE_REGISTER_COMPILER(ASTJavaCompiler, "ast_java")
.describe("AST-based compiler that produces Java code")
.set_body([](const CompilerParam& param) -> Compiler* {
    return new ASTJavaCompiler(param);
  });
}  // namespace compiler
}  // namespace treelite
