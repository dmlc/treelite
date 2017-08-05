/*!
 * Copyright 2017 by Contributors
 * \file traversal.h
 * \brief simple implementation of tree traversal, to collect branch frequencies
 * \author Philip Cho
 */
#ifndef TREELITE_TRAVERSAL_H_
#define TREELITE_TRAVERSAL_H_

#include <treelite/tree.h>
#include <dmlc/data.h>

namespace treelite {
namespace common {

template <typename IndexType>
void Traverse_(const Tree& tree, const IndexType* index,
               const dmlc::real_t* value, size_t len, int nid,
               size_t* out_counts) {
  const Tree::Node& node = tree[nid];
  
  ++out_counts[nid];
  if (!node.is_leaf()) {
    CHECK_LT(node.split_index(), std::numeric_limits<IndexType>::max());
    const IndexType split_index = static_cast<IndexType>(node.split_index());
    const tl_float threshold = node.threshold();
    const Operator op = node.comparison_op();

    // check if the instance has a value for feature (split_index)
    tl_float fvalue;
    bool is_missing = true;
    for (size_t i = 0; i < len; ++i) {
      if (index[i] == split_index) {
        is_missing = false;
        fvalue = (value == nullptr) ? 1.0 : value[i];
        break;
      }
    }
    if (is_missing) {
      Traverse_(tree, index, value, len, node.cdefault(), out_counts);
    } else {
      // perform comparison with fvalue
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
        Traverse_(tree, index, value, len, node.cleft(), out_counts);
      } else {  // right child
        Traverse_(tree, index, value, len, node.cright(), out_counts);
      }
    }
  }
}

template <typename IndexType>
void Traverse(const Tree& tree, const IndexType* index,
              const dmlc::real_t* value, size_t len,
              size_t* out_counts) {
  Traverse_(tree, index, value, len, 0, out_counts);
}

template<typename IndexType>
void ComputeBranchFrequenciesFromData(const Model& model,
                                      dmlc::Parser<IndexType>* data_parser,
                                      std::vector<size_t>* out_counts,
                                      std::vector<IndexType>* out_row_ptr,
                                      int nthread, int silent) {
  std::vector<size_t>& counts = *out_counts;
  std::vector<IndexType>& row_ptr = *out_row_ptr;
  std::vector<size_t> counts_tloc;
  row_ptr = {0};
  const size_t ntree = model.trees.size();
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);
  for (const Tree& tree : model.trees) {
    row_ptr.push_back(row_ptr.back() + tree.num_nodes);
  }
  counts.resize(row_ptr.back(), 0);
  counts_tloc.resize(row_ptr.back() * nthread, 0);

  size_t nrows_processed = 0; 
  size_t nbatch_processed = 0;
  while (data_parser->Next()) {
    const dmlc::RowBlock<IndexType>& batch = data_parser->Value();
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (size_t i = 0; i < batch.size; ++i) {
      const int tid = omp_get_thread_num();
      const IndexType off = row_ptr.back() * tid;
      const size_t ibegin = batch.offset[i];
      const size_t iend = batch.offset[i + 1];
      for (size_t tree_id = 0; tree_id < ntree; ++tree_id) {
        const Tree& tree = model.trees[tree_id];
        Traverse(tree, &batch.index[ibegin],
                    ((batch.value == nullptr) ? nullptr : &batch.value[ibegin]),
                     iend - ibegin, &counts_tloc[off + row_ptr[tree_id]]);
      }
    }
    nrows_processed += batch.size;
    ++nbatch_processed;
    if (silent > 0 && nbatch_processed % 25 == 0) {
      LOG(INFO) << nrows_processed << " rows processed";
    }
  }
  if (silent > 0) {
    LOG(INFO) << nrows_processed << " rows processed";
  }
  // perform reduction on counts
  const IndexType tot = row_ptr.back();  // # of all nodes in all trees
  for (int tid = 0; tid < nthread; ++tid) {
    const IndexType off = tot * tid;
    for (size_t i = 0; i < tot; ++i) {
      counts[i] += counts_tloc[off + i];
    }
  }
}

}  // namespace common
}  // namespace treelite

#endif  // TREELITE_TRAVERSAL_H_
