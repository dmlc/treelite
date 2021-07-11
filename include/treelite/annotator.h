/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file annotator.h
 * \author Hyunsu Cho
 * \brief Branch annotation tools
 */
#ifndef TREELITE_ANNOTATOR_H_
#define TREELITE_ANNOTATOR_H_

#include <treelite/tree.h>
#include <treelite/data.h>
#include <istream>
#include <ostream>
#include <vector>
#include <cstdio>
#include <cstdint>

namespace treelite {

/*! \brief branch annotator class */
class BranchAnnotator {
 public:
  /*!
   * \brief annotate branches in a given model using frequency patterns in the
   *        training data. The annotation can be accessed through Get() method.
   * \param model tree ensemble model
   * \param dmat training data matrix
   * \param nthread number of threads to use
   * \param verbose whether to produce extra messages
   */
  void Annotate(const Model& model, const DMatrix* dmat, int nthread, int verbose);
  /*!
   * \brief load branch annotation from a JSON file
   * \param fi input stream
   */
  void Load(std::istream& fi);
  /*!
   * \brief save branch annotation to a JSON file
   * \param fo output stream
   */
  void Save(std::ostream& fo) const;
  /*!
   * \brief fetch branch annotation.
   * Usage example:
   * \code
   *   Annotator annotator
   *   annotator.Load(fi);  // load from a stream
   *   std::vector<std::vector<size_t>> annot = annotator.Get();
   *   // access the frequency count for a specific node in a tree
   *   LOG(INFO) << "Tree " << tree_id << ", Node " << node_id << ": "
   *             << annot[tree_id][node_id];
   * \endcode
   * \return branch annotation in 2D vector
   */
  inline std::vector<std::vector<uint64_t>> Get() const {
    return counts_;
  }

 private:
  std::vector<std::vector<uint64_t>> counts_;
};

}  // namespace treelite

#endif  // TREELITE_ANNOTATOR_H_
