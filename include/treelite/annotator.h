/*!
 * Copyright (c) 2017 by Contributors
 * \file annotator.h
 * \author Philip Cho
 * \brief Branch annotation tools
 */
#ifndef TREELITE_ANNOTATOR_H_
#define TREELITE_ANNOTATOR_H_

#include <treelite/tree.h>
#include <treelite/data.h>

namespace treelite {

class BranchAnnotator {
 public:
  void Annotate(const Model& model, const DMatrix* dmat, int nthread, int verbose);
  void Load(dmlc::Stream* fi);
  void Save(dmlc::Stream* fo) const;
  inline std::vector<std::vector<size_t>> Get() const {
    return counts;
  }
 private:
  std::vector<std::vector<size_t>> counts;
};

}  // namespace treelite

#endif  // TREELITE_ANNOTATOR_H_
