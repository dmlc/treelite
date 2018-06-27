#ifndef TREELITE_COMPILER_JAVA_GET_NUM_FEATURE_H_
#define TREELITE_COMPILER_JAVA_GET_NUM_FEATURE_H_
namespace treelite {
namespace compiler {
namespace java {

inline std::string get_num_feature_func(int num_feature) {
  std::ostringstream oss;
  oss << "  public static int get_num_feature() {\n"
      << "    return " << num_feature << ";\n"
      << "  }\n";
  return oss.str();
}

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_GET_NUM_FEATURE_H_
