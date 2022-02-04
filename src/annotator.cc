/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file annotator.cc
 * \author Hyunsu Cho
 * \brief Branch annotation tools
 */

#include <treelite/logging.h>
#include <treelite/annotator.h>
#include <treelite/math.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <rapidjson/document.h>
#include <limits>
#include <thread>
#include <cstdint>
#include "threading_utils/parallel_for.h"

namespace {

using treelite::threading_utils::ThreadConfig;

template <typename ElementType>
union Entry {
  int missing;
  ElementType fvalue;
};

template <typename ElementType, typename ThresholdType, typename LeafOutputType>
void Traverse_(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
               const Entry<ElementType>* data, int nid, uint64_t* out_counts) {
  ++out_counts[nid];
  if (!tree.IsLeaf(nid)) {
    const unsigned split_index = tree.SplitIndex(nid);

    if (data[split_index].missing == -1) {
      Traverse_(tree, data, tree.DefaultChild(nid), out_counts);
    } else {
      bool result = true;
      if (tree.SplitType(nid) == treelite::SplitFeatureType::kNumerical) {
        const ThresholdType threshold = tree.Threshold(nid);
        const treelite::Operator op = tree.ComparisonOp(nid);
        const auto fvalue = static_cast<ElementType>(data[split_index].fvalue);
        result = treelite::CompareWithOp(fvalue, op, threshold);
      } else {
        const auto fvalue = data[split_index].fvalue;
        const auto matching_categories = tree.MatchingCategories(nid);
        result = (std::binary_search(matching_categories.begin(),
                                     matching_categories.end(),
                                     static_cast<uint32_t>(fvalue)));
        if (tree.CategoriesListRightChild(nid)) {
          result = !result;
        }
      }
      if (result) {  // left child
        Traverse_(tree, data, tree.LeftChild(nid), out_counts);
      } else {  // right child
        Traverse_(tree, data, tree.RightChild(nid), out_counts);
      }
    }
  }
}

template <typename ElementType, typename ThresholdType, typename LeafOutputType>
void Traverse(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
              const Entry<ElementType>* data, uint64_t* out_counts) {
  Traverse_(tree, data, 0, out_counts);
}

template <typename ElementType, typename ThresholdType, typename LeafOutputType>
inline void ComputeBranchLoopImpl(
    const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
    const treelite::DenseDMatrixImpl<ElementType>* dmat, size_t rbegin, size_t rend,
    const ThreadConfig& thread_config, const size_t* count_row_ptr, uint64_t* counts_tloc) {
  std::vector<Entry<ElementType>> inst(thread_config.nthread * dmat->num_col, {-1});
  size_t ntree = model.trees.size();
  TREELITE_CHECK_LE(rbegin, rend);
  size_t num_col = dmat->num_col;
  ElementType missing_value = dmat->missing_value;
  bool nan_missing = treelite::math::CheckNAN(missing_value);
  auto sched = treelite::threading_utils::ParallelSchedule::Static();
  treelite::threading_utils::ParallelFor(rbegin, rend, thread_config, sched,
                                         [&](std::size_t rid, int thread_id) {
    const ElementType* row = &dmat->data[rid * num_col];
    const size_t off = dmat->num_col * thread_id;
    const size_t off2 = count_row_ptr[ntree] * thread_id;
    for (size_t j = 0; j < num_col; ++j) {
      if (treelite::math::CheckNAN(row[j])) {
        TREELITE_CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        inst[off + j].fvalue = row[j];
      }
    }
    for (size_t tree_id = 0; tree_id < ntree; ++tree_id) {
      Traverse(model.trees[tree_id], &inst[off], &counts_tloc[off2 + count_row_ptr[tree_id]]);
    }
    for (size_t j = 0; j < num_col; ++j) {
      inst[off + j].missing = -1;
    }
  });
}

template <typename ElementType, typename ThresholdType, typename LeafOutputType>
inline void ComputeBranchLoopImpl(
    const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
    const treelite::CSRDMatrixImpl<ElementType>* dmat, size_t rbegin, size_t rend,
    const ThreadConfig& thread_config, const size_t* count_row_ptr, uint64_t* counts_tloc) {
  std::vector<Entry<ElementType>> inst(thread_config.nthread * dmat->num_col, {-1});
  size_t ntree = model.trees.size();
  TREELITE_CHECK_LE(rbegin, rend);
  auto sched = treelite::threading_utils::ParallelSchedule::Static();
  treelite::threading_utils::ParallelFor(rbegin, rend, thread_config, sched,
                                         [&](std::size_t rid, int thread_id) {
    const size_t off = dmat->num_col * thread_id;
    const size_t off2 = count_row_ptr[ntree] * thread_id;
    const size_t ibegin = dmat->row_ptr[rid];
    const size_t iend = dmat->row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].fvalue = dmat->data[i];
    }
    for (size_t tree_id = 0; tree_id < ntree; ++tree_id) {
      Traverse(model.trees[tree_id], &inst[off], &counts_tloc[off2 + count_row_ptr[tree_id]]);
    }
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].missing = -1;
    }
  });
}

template <typename ElementType>
class ComputeBranchLoopDispatcherWithDenseDMatrix {
 public:
  template <typename ThresholdType, typename LeafOutputType>
  inline static void Dispatch(
      const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
      const treelite::DMatrix* dmat, size_t rbegin, size_t rend, const ThreadConfig& thread_config,
      const size_t* count_row_ptr, uint64_t* counts_tloc) {
    const auto* dmat_ = static_cast<const treelite::DenseDMatrixImpl<ElementType>*>(dmat);
    TREELITE_CHECK(dmat_) << "Dangling data matrix reference detected";
    ComputeBranchLoopImpl(model, dmat_, rbegin, rend, thread_config, count_row_ptr, counts_tloc);
  }
};

template <typename ElementType>
class ComputeBranchLoopDispatcherWithCSRDMatrix {
 public:
  template <typename ThresholdType, typename LeafOutputType>
  inline static void Dispatch(
      const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
      const treelite::DMatrix* dmat, size_t rbegin, size_t rend, const ThreadConfig& thread_config,
      const size_t* count_row_ptr, uint64_t* counts_tloc) {
    const auto* dmat_ = static_cast<const treelite::CSRDMatrixImpl<ElementType>*>(dmat);
    TREELITE_CHECK(dmat_) << "Dangling data matrix reference detected";
    ComputeBranchLoopImpl(model, dmat_, rbegin, rend, thread_config, count_row_ptr, counts_tloc);
  }
};

template <typename ThresholdType, typename LeafOutputType>
inline void ComputeBranchLoop(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                              const treelite::DMatrix* dmat, size_t rbegin,
                              size_t rend, const ThreadConfig& thread_config,
                              const size_t* count_row_ptr, uint64_t* counts_tloc) {
  switch (dmat->GetType()) {
  case treelite::DMatrixType::kDense: {
    treelite::DispatchWithTypeInfo<ComputeBranchLoopDispatcherWithDenseDMatrix>(
        dmat->GetElementType(), model, dmat, rbegin, rend, thread_config, count_row_ptr,
        counts_tloc);
    break;
  }
  case treelite::DMatrixType::kSparseCSR: {
    treelite::DispatchWithTypeInfo<ComputeBranchLoopDispatcherWithCSRDMatrix>(
        dmat->GetElementType(), model, dmat, rbegin, rend, thread_config, count_row_ptr,
        counts_tloc);
    break;
  }
  default:
    TREELITE_LOG(FATAL)
      << "Annotator does not support DMatrix of type " << static_cast<int>(dmat->GetType());
    break;
  }
}

}  // anonymous namespace

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
inline void
AnnotateImpl(
    const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
    const treelite::DMatrix* dmat, int nthread, int verbose,
    std::vector<std::vector<uint64_t>>* out_counts) {
  std::vector<uint64_t> new_counts;
  std::vector<uint64_t> counts_tloc;
  std::vector<size_t> count_row_ptr;

  count_row_ptr = {0};
  const size_t ntree = model.trees.size();
  ThreadConfig thread_config = threading_utils::ConfigureThreadConfig(nthread);
  for (const treelite::Tree<ThresholdType, LeafOutputType>& tree : model.trees) {
    count_row_ptr.push_back(count_row_ptr.back() + tree.num_nodes);
  }
  new_counts.resize(count_row_ptr[ntree], 0);
  counts_tloc.resize(count_row_ptr[ntree] * thread_config.nthread, 0);

  const size_t num_row = dmat->GetNumRow();
  const size_t pstep = (num_row + 19) / 20;
  // interval to display progress
  for (size_t rbegin = 0; rbegin < num_row; rbegin += pstep) {
    const size_t rend = std::min(rbegin + pstep, num_row);
    ComputeBranchLoop(model, dmat, rbegin, rend, thread_config, &count_row_ptr[0],
                      &counts_tloc[0]);
    if (verbose > 0) {
      TREELITE_LOG(INFO) << rend << " of " << num_row << " rows processed";
    }
  }

  // perform reduction on counts
  for (std::uint32_t tid = 0; tid < thread_config.nthread; ++tid) {
    const size_t off = count_row_ptr[ntree] * tid;
    for (size_t i = 0; i < count_row_ptr[ntree]; ++i) {
      new_counts[i] += counts_tloc[off + i];
    }
  }

  // change layout of counts
  std::vector<std::vector<uint64_t>>& counts = *out_counts;
  for (size_t i = 0; i < ntree; ++i) {
    counts.emplace_back(&new_counts[count_row_ptr[i]], &new_counts[count_row_ptr[i + 1]]);
  }
}

void
BranchAnnotator::Annotate(const Model& model, const DMatrix* dmat, int nthread, int verbose) {
  TypeInfo threshold_type = model.GetThresholdType();
  model.Dispatch([this, dmat, nthread, verbose, threshold_type](auto& handle) {
    AnnotateImpl(handle, dmat, nthread, verbose, &this->counts_);
  });
}

void
BranchAnnotator::Load(std::istream& fi) {
  rapidjson::IStreamWrapper is(fi);

  rapidjson::Document doc;
  doc.ParseStream(is);

  std::string err_msg = "JSON file must contain a list of lists of integers";
  TREELITE_CHECK(doc.IsArray()) << err_msg;
  counts_.clear();
  for (const auto& node_cnt : doc.GetArray()) {
    TREELITE_CHECK(node_cnt.IsArray()) << err_msg;
    counts_.emplace_back();
    for (const auto& e : node_cnt.GetArray()) {
      counts_.back().push_back(e.GetUint64());
    }
  }
}

void
BranchAnnotator::Save(std::ostream& fo) const {
  rapidjson::OStreamWrapper os(fo);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(os);

  writer.StartArray();
  for (const auto& node_cnt : counts_) {
    writer.StartArray();
    for (auto e : node_cnt) {
      writer.Uint64(e);
    }
    writer.EndArray();
  }
  writer.EndArray();
}

}  // namespace treelite
