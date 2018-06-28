/*!
 * Copyright 2017 by Contributors
 * \file param.h
 * \brief Parameters for tree compiler
 * \author Philip Cho
 */
#ifndef TREELITE_COMPILER_PARAM_H_
#define TREELITE_COMPILER_PARAM_H_

#include <dmlc/parameter.h>
#include <string>

namespace treelite {
namespace compiler {

/*! \brief parameters for tree compiler */
struct CompilerParam : public dmlc::Parameter<CompilerParam> {
  /*!
  * \defgroup compiler_param
  * parameters for tree compiler
  * \{
  */
  /*! \brief name of model annotation file. Use the class ``treelite.Annotator``
      to generate this file.  */
  std::string annotate_in;
  /*! \brief whether to quantize threshold points (0: no, >0: yes) */
  int quantize;
  /*! \brief option to enable parallel compilation;
             if set to nonzero, the trees will be evely distributed
             into ``[parallel_comp]`` files. Set this option to improve
             compilation time and reduce memory consumption during
             compilation. */
  int parallel_comp;
  /*! \brief if >0, produce extra messages */
  int verbose;
  /*! \} */
  int max_unit_size;
  /*! \brief native lib name (without extension) */
  std::string native_lib_name;
  /*! \brief java package name */
  std::string java_package;
  /*! \brief java file path prefix */
  std::string java_file_prefix;

  // declare parameters
  DMLC_DECLARE_PARAMETER(CompilerParam) {
    DMLC_DECLARE_FIELD(annotate_in).set_default("NULL")
      .describe("Name of model annotation file");
    DMLC_DECLARE_FIELD(quantize).set_lower_bound(0).set_default(0)
      .describe("whether to quantize threshold points (0: no, >0: yes)");
    DMLC_DECLARE_FIELD(parallel_comp).set_lower_bound(0).set_default(0)
      .describe("option to enable parallel compilation;"
                "if set to nonzero, the trees will be evely distributed"
                "into [parallel_comp] files.");
    DMLC_DECLARE_FIELD(verbose).set_default(0)
      .describe("if >0, produce extra messages");
    DMLC_DECLARE_FIELD(max_unit_size).set_default(100).set_lower_bound(5);
    DMLC_DECLARE_FIELD(native_lib_name).set_default("predictor");
    DMLC_DECLARE_FIELD(java_package).set_default("treelite.predictor");
    DMLC_DECLARE_FIELD(java_file_prefix).set_default("src/main/java/treelite/predictor/");
  }
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_PARAM_H_
