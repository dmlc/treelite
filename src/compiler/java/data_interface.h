/*!
 * Copyright (c) 2018 by Contributors
 * \file data_interface.h
 * \author Philip Cho
 * \brief template for Java data interface
 */

#ifndef TREELITE_COMPILER_JAVA_DATA_INTERFACE_H_
#define TREELITE_COMPILER_JAVA_DATA_INTERFACE_H_

namespace treelite {
namespace compiler {
namespace java {

const char* data_interface =
R"TREELITETEMPLATE(
package ml.dmlc.treelite;

public interface Data {
  public void setFValue(float val);
  public void setMissing();
  public boolean isMissing();
  public float getFValue();
}
)TREELITETEMPLATE";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_DATA_INTERFACE_H_
