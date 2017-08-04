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
  /*! \brief name of model annotation file */
  std::string annotate_in;
  /*! \brief whether to quantize threshold points (0: no, >0: yes) */
  int quantize;
  /*! \brief option to distribute compiled trees into different files;
             set to nonzero to specify the number of trees each file
             should contain */
  int dist_comp;

  // declare parameters
  DMLC_DECLARE_PARAMETER(CompilerParam) {
    DMLC_DECLARE_FIELD(annotate_in).set_default("NULL")
      .describe("Name of model annotation file");
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
