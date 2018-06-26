#ifndef TREELITE_COMPILER_JAVA_GET_NUM_FEATURE_H_
#define TREELITE_COMPILER_JAVA_GET_NUM_FEATURE_H_
inline std::string get_num_feature_func(int num_feature) {
  std::ostringstream oss;
  oss << "  public static int get_num_feature() {\n"
      << "    return " << num_feature << ";\n"
      << "  }\n";
  return oss.str();
}
#endif  // TREELITE_COMPILER_JAVA_GET_NUM_FEATURE_H_
