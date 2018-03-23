inline std::string get_num_output_group_func(int num_output_group) {
  std::ostringstream oss;
  oss << "  public static int get_num_output_group() {\n"
      << "    return " << num_output_group << ";\n"
      << "  }\n";
  return oss.str();
}