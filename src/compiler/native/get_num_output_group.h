#ifndef TREELITE_COMPILER_NATIVE_GET_NUM_OUTPUT_GROUP_H_
#define TREELITE_COMPILER_NATIVE_GET_NUM_OUTPUT_GROUP_H_
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
#endif  // TREELITE_COMPILER_NATIVE_GET_NUM_OUTPUT_GROUP_H_
