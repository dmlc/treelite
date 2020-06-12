/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file annotator.cc
 * \author Hyunsu Cho
 * \brief Branch annotation tools
 */

#include <treelite/annotator.h>
#include <treelite/omp.h>
#include <dmlc/json.h>
#include <limits>
#include <cstdint>

namespace {

union Entry {
  int missing;
  float fvalue;
};

void Traverse_(const treelite::Tree& tree, const Entry* data,
               int nid, size_t* out_counts) {
  ++out_counts[nid];
  if (!tree.IsLeaf(nid)) {
    const unsigned split_index = tree.SplitIndex(nid);

    if (data[split_index].missing == -1) {
      Traverse_(tree, data, tree.DefaultChild(nid), out_counts);
    } else {
      bool result = true;
      if (tree.SplitType(nid) == treelite::SplitFeatureType::kNumerical) {
        const treelite::tl_float threshold = tree.Threshold(nid);
        const treelite::Operator op = tree.ComparisonOp(nid);
        const auto fvalue = static_cast<treelite::tl_float>(data[split_index].fvalue);
        result = treelite::CompareWithOp(fvalue, op, threshold);
      } else {
        const auto fvalue = data[split_index].fvalue;
        const auto left_categories = tree.LeftCategories(nid);
        result = (std::binary_search(left_categories.begin(),
                                     left_categories.end(),
                                     static_cast<uint32_t>(fvalue)));
      }
      if (result) {  // left child
        Traverse_(tree, data, tree.LeftChild(nid), out_counts);
      } else {  // right child
        Traverse_(tree, data, tree.RightChild(nid), out_counts);
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
  CHECK_LE(rbegin, rend);
  CHECK_LT(static_cast<int64_t>(rend), std::numeric_limits<int64_t>::max());
  const auto rbegin_i = static_cast<int64_t>(rbegin);
  const auto rend_i = static_cast<int64_t>(rend);
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (int64_t rid = rbegin_i; rid < rend_i; ++rid) {
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

}  // anonymous namespace

namespace treelite {

void
BranchAnnotator::Annotate(const Model& model, const DMatrix* dmat,
                          int nthread, int verbose) {
  std::vector<size_t> new_counts;
  std::vector<size_t> counts_tloc;
  std::vector<size_t> count_row_ptr;
  count_row_ptr = {0};
  const size_t ntree = model.trees.size();
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);
  for (const Tree& tree : model.trees) {
    count_row_ptr.push_back(count_row_ptr.back() + tree.num_nodes);
  }
  new_counts.resize(count_row_ptr[ntree], 0);
  counts_tloc.resize(count_row_ptr[ntree] * nthread, 0);

  std::vector<Entry> inst(nthread * dmat->num_col, {-1});
  const size_t pstep = (dmat->num_row + 19) / 20;
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
      new_counts[i] += counts_tloc[off + i];
    }
  }

  // change layout of counts
  for (size_t i = 0; i < ntree; ++i) {
    this->counts.emplace_back(&new_counts[count_row_ptr[i]],
                              &new_counts[count_row_ptr[i + 1]]);
  }
}

void
BranchAnnotator::Load(dmlc::Stream* fi) {
  dmlc::istream is(fi);
  std::unique_ptr<dmlc::JSONReader> reader(new dmlc::JSONReader(&is));
  reader->Read(&counts);
}

void
BranchAnnotator::Save(dmlc::Stream* fo) const {
  dmlc::ostream os(fo);
  std::unique_ptr<dmlc::JSONWriter> writer(new dmlc::JSONWriter(&os));
  writer->Write(counts);
}

}  // namespace treelite
