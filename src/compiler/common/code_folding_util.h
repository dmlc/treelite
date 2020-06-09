/*!
 * Copyright (c) 2018 by Contributors
 * \file code_folding_util.h
 * \author Philip Cho
 * \brief Utilities for code folding
 */
#ifndef TREELITE_COMPILER_COMMON_CODE_FOLDING_UTIL_H_
#define TREELITE_COMPILER_COMMON_CODE_FOLDING_UTIL_H_

#include <queue>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include <dmlc/logging.h>
#include <fmt/format.h>
#include "../ast/ast.h"
#include "./format_util.h"
#include "./categorical_bitmap.h"

using namespace fmt::literals;

namespace treelite {
namespace compiler {
namespace common_util {

template <typename OutputFormatFunc>
inline void
RenderCodeFolderArrays(const CodeFolderNode* node,
                       bool quantize,
                       bool use_boolean_literal,
                       const char* node_entry_template,
                       OutputFormatFunc RenderOutputStatement,
                       std::string* array_nodes,
                       std::string* array_cat_bitmap,
                       std::string* array_cat_begin,
                       std::string* output_switch_statements,
                       Operator* common_comp_op) {
  CHECK_EQ(node->children.size(), 1);
  const int tree_id = node->children[0]->tree_id;
  // list of descendants, with newly assigned ID's
  std::unordered_map<ASTNode*, int> descendants;
  // list of all OutputNode's among the descendants
  std::vector<OutputNode*> output_nodes;
  // two arrays used to store categorical split info
  std::vector<uint64_t> cat_bitmap;
  std::vector<size_t> cat_begin{0};

  // 1. Assign new continuous node ID's (0, 1, 2, ...) by traversing the
  // subtree breadth-first
  {
    std::queue<ASTNode*> Q;
    std::set<treelite::Operator> ops;
    int new_node_id = 0;
    int new_leaf_id = -1;
    Q.push(node->children[0]);
    while (!Q.empty()) {
      ASTNode* e = Q.front(); Q.pop();
      // sanity check: all descendants must have same tree_id
      CHECK_EQ(e->tree_id, tree_id);
      // sanity check: all descendants must be ConditionNode or OutputNode
      ConditionNode* t1 = dynamic_cast<ConditionNode*>(e);
      OutputNode* t2 = dynamic_cast<OutputNode*>(e);
      NumericalConditionNode* t3;
      CHECK(t1 || t2);
      if (t2) {  // e is OutputNode
        descendants[e] = new_leaf_id--;
      } else {
        if ( (t3 = dynamic_cast<NumericalConditionNode*>(t1)) ) {
          ops.insert(t3->op);
        }
        descendants[e] = new_node_id++;
      }
      for (ASTNode* child : e->children) {
        Q.push(child);
      }
    }
    // sanity check: all numerical splits must have identical comparison operators
    CHECK_LE(ops.size(), 1);
    *common_comp_op = ops.empty() ? Operator::kLT : *ops.begin();
  }

  // 2. Render node_treeXX_nodeXX[] by traversing the subtree once again.
  // Now we can use the re-assigned node ID's.
  {
    ArrayFormatter formatter(80, 2);

    bool default_left;
    std::string threshold;
    int left_child_id, right_child_id;
    unsigned int split_index;
    OutputNode* t1;
    NumericalConditionNode* t2;
    CategoricalConditionNode* t3;

    std::queue<ASTNode*> Q;
    Q.push(node->children[0]);
    while (!Q.empty()) {
      ASTNode* e = Q.front(); Q.pop();
      if ( (t1 = dynamic_cast<OutputNode*>(e)) ) {
        output_nodes.push_back(t1);
        // don't render OutputNode but save it for later
      } else {
        CHECK_EQ(e->children.size(), 2U);
        left_child_id = descendants[ e->children[0] ];
        right_child_id = descendants[ e->children[1] ];
        if ( (t2 = dynamic_cast<NumericalConditionNode*>(e)) ) {
          default_left = t2->default_left;
          split_index = t2->split_index;
          threshold
           = quantize ? std::to_string(t2->threshold.int_val)
                      : ToStringHighPrecision(t2->threshold.float_val);
        } else {
          CHECK((t3 = dynamic_cast<CategoricalConditionNode*>(e)));
          default_left = t3->default_left;
          split_index = t3->split_index;
          threshold = "-1";  // dummy value
          CHECK(!t3->convert_missing_to_zero)
            << "Code folding not supported, because a categorical split "
            << "is supposed to convert missing values into zeros, and this "
            << "is not possible with current code folding implementation.";
          std::vector<uint64_t> bitmap
            = GetCategoricalBitmap(t3->left_categories);
          cat_bitmap.insert(cat_bitmap.end(), bitmap.begin(), bitmap.end());
          cat_begin.push_back(cat_bitmap.size());
        }
        const char* (*BoolWrapper)(bool);
        if (use_boolean_literal) {
          BoolWrapper = [](bool x) { return x ? "true" : "false"; };
        } else {
          BoolWrapper = [](bool x) { return x ? "1" : "0"; };
        }
        formatter << fmt::format(node_entry_template,
                                  "default_left"_a = BoolWrapper(default_left),
                                  "split_index"_a = split_index,
                                  "threshold"_a = threshold,
                                  "left_child"_a = left_child_id,
                                  "right_child"_a = right_child_id);
      }
      for (ASTNode* child : e->children) {
        Q.push(child);
      }
    }
    *array_nodes = formatter.str();
  }
  // 3. render cat_bitmap_treeXX_nodeXX[] and cat_begin_treeXX_nodeXX[]
  if (cat_bitmap.empty()) {  // do not render empty arrays
    *array_cat_bitmap = "";
    *array_cat_begin = "";
  } else {
    {
      ArrayFormatter formatter(80, 2);
      for (uint64_t e : cat_bitmap) {
        formatter << fmt::format("{:#X}", e);
      }
      *array_cat_bitmap = formatter.str();
    }
    {
      ArrayFormatter formatter(80, 2);
      for (size_t e : cat_begin) {
        formatter << e;
      }
      *array_cat_begin = formatter.str();
    }
  }
  // 4. Render switch statement to associate each node ID with an output
  *output_switch_statements = "switch (nid) {\n";
  for (OutputNode* e : output_nodes) {
    const int node_id = descendants[static_cast<ASTNode*>(e)];
    *output_switch_statements
      += fmt::format(" case {node_id}:\n"
                      "{output_statement}"
                      "  break;\n",
            "node_id"_a = node_id,
            "output_statement"_a = IndentMultiLineString(RenderOutputStatement(e), 2));
  }
  *output_switch_statements += "}\n";
}

}  // namespace common_util
}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_COMMON_CODE_FOLDING_UTIL_H_
