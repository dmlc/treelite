/*!
 * Copyright (c) 2017 by Contributors
 * \file entry_type.h
 * \author Philip Cho
 * \brief template for Entry type in generated Java code
 */
#ifndef TREELITE_COMPILER_JAVA_ENTRY_TYPE_H_
#define TREELITE_COMPILER_JAVA_ENTRY_TYPE_H_
#include <string>

namespace treelite {
namespace compiler {
namespace java {

inline std::string entry_type(const std::string& java_package) {
  return
    "package " + java_package + ";\n" +
    "\n" +
    "import javolution.io.Union;\n" +
    "\n" +
    "public class Entry extends Union {\n" +
    "  public Signed32 missing = new Signed32();\n" +
    "  public Float32  fvalue  = new Float32();\n" +
    "  public Signed32 qvalue  = new Signed32();\n" +
    "}\n";
}

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_ENTRY_TYPE_H_
