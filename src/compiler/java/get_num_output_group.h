#ifndef TREELITE_COMPILER_JAVA_GET_NUM_OUTPUT_GROUP_H_
#define TREELITE_COMPILER_JAVA_GET_NUM_OUTPUT_GROUP_H_
namespace treelite {
namespace compiler {
namespace java {

inline std::string get_num_output_group_func(int num_output_group) {
  std::ostringstream oss;
  oss << "  public static int get_num_output_group() {\n"
      << "    return " << num_output_group << ";\n"
      << "  }\n";
  return oss.str();
}

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_GET_NUM_OUTPUT_GROUP_H_
