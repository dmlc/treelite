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
  /*! \brief how test instances should be accessed */
  int data_layout;
  /*! \brief name of model annotation file */
  std::string annotate_in;
  /*! \brief size of batch for batching prediction function */
  int batch_size;
  /*! \brief whether to quantize threshold points (0: no, >0: yes) */
  int quantize;
  /*! \brief option to distribute compiled trees into different files;
             set to nonzero to specify the number of trees each file
             should contain */
  int dist_comp;

  // declare parameters
  DMLC_DECLARE_PARAMETER(CompilerParam) {
    DMLC_DECLARE_FIELD(data_layout).set_default(0)
      .describe("how test instances should be accessed")
      .add_enum("dense", 0).add_enum("compressed", 1).add_enum("sparse", 2);
    DMLC_DECLARE_FIELD(annotate_in).set_default("NULL")
      .describe("Name of model annotation file");
    DMLC_DECLARE_FIELD(batch_size).set_lower_bound(1).set_default(1)
      .describe("Batch size for batching prediction function");
    DMLC_DECLARE_FIELD(quantize).set_lower_bound(0).set_default(0)
      .describe("whether to quantize threshold points (0: no, >0: yes)");
    DMLC_DECLARE_FIELD(dist_comp).set_lower_bound(0).set_default(0)
      .describe("option to distribute compiled trees into different files; "
                "set to nonzero to specify the number of trees each file "
                "should contain");
  }
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_PARAM_H_
