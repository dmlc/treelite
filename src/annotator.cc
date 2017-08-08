/*!
 * Copyright (c) 2017 by Contributors
 * \file annotator.cc
 * \author Philip Cho
 * \brief Branch annotation tools
 */
 
#include <treelite/annotator.h>
#include <omp.h>

namespace {

union Entry {
  int missing;
  float fvalue;
};

void Traverse_(const treelite::Tree& tree, const Entry* data,
               int nid, size_t* out_counts) {
  const treelite::Tree::Node& node = tree[nid];
  
  ++out_counts[nid];
  if (!node.is_leaf()) {
    const unsigned split_index = node.split_index();
    const treelite::tl_float threshold = node.threshold();
    const treelite::Operator op = node.comparison_op();

    if (data[split_index].missing == -1) {
      Traverse_(tree, data, node.cdefault(), out_counts);
    } else {
      // perform comparison with fvalue
      const treelite::tl_float fvalue
        = static_cast<treelite::tl_float>(data[split_index].fvalue);
      bool result = true;
      switch (op) {
       case treelite::Operator::kEQ:
        result = (fvalue == threshold); break;
       case treelite::Operator::kLT:
        result = (fvalue <  threshold); break;
       case treelite::Operator::kLE:
        result = (fvalue <= threshold); break;
       case treelite::Operator::kGT:
        result = (fvalue >  threshold); break;
       case treelite::Operator::kGE:
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

void Traverse(const treelite::Tree& tree, const Entry* data,
              size_t* out_counts) {
  Traverse_(tree, data, 0, out_counts);
}

inline void ComputeBranchLoop(const treelite::Model& model,
                              const treelite::DMatrix* dmat,
                              size_t rbegin, size_t rend, int nthread,
                              const size_t* count_row_ptr,
                              size_t* counts_tloc, Entry* inst) {
  const size_t ntree = model.trees.size();
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (size_t rid = rbegin; rid < rend; ++rid) {
    const int tid = omp_get_thread_num();
    const size_t off = dmat->num_col * tid;
    const size_t off2 = count_row_ptr[ntree] * tid;
    const size_t ibegin = dmat->row_ptr[rid];
    const size_t iend = dmat->row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].fvalue = dmat->data[i];
    }
    for (size_t tree_id = 0; tree_id < ntree; ++tree_id) {
      Traverse(model.trees[tree_id], &inst[off],
               &counts_tloc[off2 + count_row_ptr[tree_id]]);
    }
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].missing = -1;
    }
  }
}

}  // namespace anonymous

namespace treelite {

void
BranchAnnotator::Annotate(const Model& model, const DMatrix* dmat,
                          int nthread, int verbose) {
  std::vector<size_t> counts;
  std::vector<size_t> counts_tloc;
  std::vector<size_t> count_row_ptr;
  count_row_ptr = {0};
  const size_t ntree = model.trees.size();
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);
  for (const Tree& tree : model.trees) {
    count_row_ptr.push_back(count_row_ptr.back() + tree.num_nodes);
  }
  counts.resize(count_row_ptr[ntree], 0);
  counts_tloc.resize(count_row_ptr[ntree] * nthread, 0);
  
  std::vector<Entry> inst(nthread * dmat->num_col, {-1});
  const size_t pstep = (dmat->num_row + 99) / 100;
      // interval to display progress
  for (size_t rbegin = 0; rbegin < dmat->num_row; rbegin += pstep) {
    const size_t rend = std::min(rbegin + pstep, dmat->num_row);
    ComputeBranchLoop(model, dmat, rbegin, rend, nthread,
                      &count_row_ptr[0], &counts_tloc[0], &inst[0]);
    if (verbose > 0) {
      LOG(INFO) << rend << " of " << dmat->num_row << " rows processed";
    }
  }

  // perform reduction on counts
  for (int tid = 0; tid < nthread; ++tid) {
    const size_t off = count_row_ptr[ntree] * tid;
    for (size_t i = 0; i < count_row_ptr[ntree]; ++i) {
      counts[i] += counts_tloc[off + i];
    }
  }
  
  // change layout of counts
  for (size_t i = 0; i < ntree; ++i) {
    this->counts.emplace_back(&counts[count_row_ptr[i]],
                              &counts[count_row_ptr[i + 1]]);
  }
}

void
BranchAnnotator::Load(dmlc::Stream* fi) {
  dmlc::istream is(fi);
  auto reader = common::make_unique<dmlc::JSONReader>(&is);
  reader->Read(&counts);
}

void
BranchAnnotator::Save(dmlc::Stream* fo) const {
  dmlc::ostream os(fo);
  auto writer = common::make_unique<dmlc::JSONWriter>(&os);
  writer->Write(counts);
}

}  // namespace treelite
