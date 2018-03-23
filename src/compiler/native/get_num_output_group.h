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