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

const char* entry_type_template =
R"TREELITETEMPLATE(
package {java_package};

import javolution.io.Union;

public class Entry extends Union {{
  public Signed32 missing = new Signed32();
  public Float32  fvalue  = new Float32();
  public Signed32 qvalue  = new Signed32();
}}
)TREELITETEMPLATE";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_ENTRY_TYPE_H_
