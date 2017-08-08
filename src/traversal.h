/*!
 * Copyright 2017 by Contributors
 * \file traversal.h
 * \brief simple implementation of tree traversal, to collect branch frequencies
 * \author Philip Cho
 */
#ifndef TREELITE_TRAVERSAL_H_
#define TREELITE_TRAVERSAL_H_

#include <treelite/tree.h>
#include <treelite/data.h>
#include <dmlc/data.h>
#include <omp.h>

namespace treelite {
namespace common {

union Entry {
  int missing;
  float fvalue;
};

void Traverse_(const Tree& tree, const Entry* data,
               int nid, size_t* out_counts) {
  const Tree::Node& node = tree[nid];
  
  ++out_counts[nid];
  if (!node.is_leaf()) {
    const unsigned split_index = node.split_index();
    const tl_float threshold = node.threshold();
    const Operator op = node.comparison_op();

    if (data[split_index].missing == -1) {
      Traverse_(tree, data, node.cdefault(), out_counts);
    } else {
      // perform comparison with fvalue
      const tl_float fvalue = static_cast<tl_float>(data[split_index].fvalue);
      bool result = true;
      switch (op) {
       case Operator::kEQ:
        result = (fvalue == threshold); break;
       case Operator::kLT:
        result = (fvalue <  threshold); break;
       case Operator::kLE:
        result = (fvalue <= threshold); break;
       case Operator::kGT:
        result = (fvalue >  threshold); break;
       case Operator::kGE:
        result = (fvalue >= threshold); break;
       default:
        LOG(FATAL) << "operator undefined";
      }
      if (result) {  // left child
        Traverse_(tree, data, node.cleft(), out_counts);
      } else {  // right child
        Traverse_(tree, data, node.cright(), out_counts);
      }
    }
  }
}

void Traverse(const Tree& tree, const Entry* data, size_t* out_counts) {
  Traverse_(tree, data, 0, out_counts);
}

static inline void ComputeBranchLoop(const Model& model, const DMatrix& dmat,
                                     size_t rbegin, size_t rend, int nthread,
                                     std::vector<size_t>& count_row_ptr,
                                     std::vector<size_t>& counts_tloc,
                                     std::vector<Entry>& inst) {
  const size_t ntree = model.trees.size();
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (size_t rid = rbegin; rid < rend; ++rid) {
    const int tid = omp_get_thread_num();
    const size_t off = dmat.num_col * tid;
    const size_t off2 = count_row_ptr.back() * tid;
    const size_t ibegin = dmat.row_ptr[rid];
    const size_t iend = dmat.row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat.col_ind[i]].fvalue = dmat.data[i];
    }
    for (size_t tree_id = 0; tree_id < ntree; ++tree_id) {
      Traverse(model.trees[tree_id], &inst[off],
               &counts_tloc[off2 + count_row_ptr[tree_id]]);
    }
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat.col_ind[i]].missing = -1;
    }
  }
}

void ComputeBranchFrequenciesFromData(const Model& model,
                                      const DMatrix& dmat,
                                      std::vector<size_t>* out_counts,
                                      std::vector<size_t>* out_row_ptr,
                                      int nthread, int verbose) {
  std::vector<size_t>& counts = *out_counts;
  std::vector<size_t>& count_row_ptr = *out_row_ptr;
  std::vector<size_t> counts_tloc;
  count_row_ptr = {0};
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);
  for (const Tree& tree : model.trees) {
    count_row_ptr.push_back(count_row_ptr.back() + tree.num_nodes);
  }
  counts.resize(count_row_ptr.back(), 0);
  counts_tloc.resize(count_row_ptr.back() * nthread, 0);
  
  std::vector<Entry> inst(nthread * dmat.num_col, {-1});
  const size_t pstep = (dmat.num_row + 99) / 100;  // interval to display progress
  for (size_t rbegin = 0; rbegin < dmat.num_row; rbegin += pstep) {
    const size_t rend = std::min(rbegin + pstep, dmat.num_row);
    ComputeBranchLoop(model, dmat, rbegin, rend, nthread,
                      count_row_ptr, counts_tloc, inst);
    if (verbose > 0) {
      LOG(INFO) << rend << " of " << dmat.num_row << " rows processed";
    }
  }

  // perform reduction on counts
  const size_t tot = count_row_ptr.back();  // # of all nodes in all trees
  for (int tid = 0; tid < nthread; ++tid) {
    const size_t off = tot * tid;
    for (size_t i = 0; i < tot; ++i) {
      counts[i] += counts_tloc[off + i];
    }
  }
}

}  // namespace common
}  // namespace treelite

#endif  // TREELITE_TRAVERSAL_H_
