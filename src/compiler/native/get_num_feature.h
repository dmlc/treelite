inline std::string get_num_feature_func(int num_feature) {
  std::ostringstream oss;
  oss << "int get_num_feature() {\n"
      << "  return " << num_feature << ";\n"
      << "}\n";
  return oss.str();
}

inline std::string get_num_feature_func_prototype() {
  return "int get_num_feature();\n";
}