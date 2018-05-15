/*!
 * Copyright 2017 by Contributors
 * \file param.h
 * \brief Parameters for tree compiler
 * \author Philip Cho
 */
#ifndef TREELITE_PARAM_H_
#define TREELITE_PARAM_H_

#include <dmlc/parameter.h>

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
  /*! \brief parameter to conform to Java's 64K bytecode limit.
             Not applicable to native target */
  int max_unit_size;
  /*! \brief java package name */
  std::string java_package;
  /*! \brief java package version */
  std::string java_package_version;
  /*! \brief java file path prefix */
  std::string java_file_prefix;
  /*! \brief parameter for folding rarely visited subtrees (no if/else blocks);
             all nodes whose data counts are lower than that of the root node
             of the decision tree by [code_folding_data_count_req] will be
             folded. To diable folding, set to +inf. */
  double code_folding_data_count_req;
  /*! \brief parameter for folding rarely visited subtrees (no if/else blocks);
             all nodes whose hessian sums are lower than that of the root node
             of the decision tree by [code_folding_sum_hess_req] will be
             folded. To disable folding, set to +inf. */
  double code_folding_sum_hess_req;
  /*! \brief path to save a dump of AST. If NULL, don't generate dump */
  std::string ast_dump_path;
  /*! \brief whether AST dump should be binary (>0) or human-readable text (<=0) */
  int ast_dump_binary;
  /*! \} */

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
    DMLC_DECLARE_FIELD(java_package).set_default("treelite.predictor");
    DMLC_DECLARE_FIELD(java_package_version).set_default("1.0-SNAPSHOT");
    DMLC_DECLARE_FIELD(java_file_prefix)
       .set_default("src/main/java/treelite/predictor/");
    DMLC_DECLARE_FIELD(code_folding_data_count_req)
       .set_default(std::numeric_limits<double>::infinity())
       .set_lower_bound(0);
    DMLC_DECLARE_FIELD(code_folding_sum_hess_req)
       .set_default(std::numeric_limits<double>::infinity())
       .set_lower_bound(0);
    DMLC_DECLARE_FIELD(ast_dump_path)
       .set_default("NULL")
       .describe("Path to save a dump of AST");
    DMLC_DECLARE_FIELD(ast_dump_binary)
       .set_default(1);
  }
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_PARAM_H_
