/*!
 * Copyright (c) 2017 by Contributors
 * \file get_num_output_group.h
 * \author Philip Cho
 * \brief template for get_num_output_group() function in generated C code
 */
#ifndef TREELITE_COMPILER_NATIVE_GET_NUM_OUTPUT_GROUP_H_
#define TREELITE_COMPILER_NATIVE_GET_NUM_OUTPUT_GROUP_H_

#include <string>

namespace treelite {
namespace compiler {
namespace native {

inline std::string get_num_output_group_func(int num_output_group) {
  std::ostringstream oss;
  oss << "size_t get_num_output_group(void) {\n"
      << "  return " << num_output_group << ";\n"
      << "}\n";
  return oss.str();
}

inline std::string get_num_output_group_func_prototype() {
  return "size_t get_num_output_group(void);\n";
}

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_GET_NUM_OUTPUT_GROUP_H_
