/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file data.cc
 * \author Hyunsu Cho
 * \brief Input data structure of Treelite
 */

#include <treelite/data.h>
#include <treelite/omp.h>
#include <memory>
#include <limits>
#include <cstdint>

namespace treelite {

LegacyDMatrix*
LegacyDMatrix::Create(const char* filename, const char* format,
                      int nthread, int verbose) {
  std::unique_ptr<dmlc::Parser<uint32_t>> parser(
    dmlc::Parser<uint32_t>::Create(filename, 0, 1, format));
  return Create(parser.get(), nthread, verbose);
}

LegacyDMatrix*
LegacyDMatrix::Create(dmlc::Parser<uint32_t>* parser, int nthread, int verbose) {
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);

  LegacyDMatrix* dmat = new LegacyDMatrix();
  dmat->Clear();
  auto& data_ = dmat->data;
  auto& col_ind_ = dmat->col_ind;
  auto& row_ptr_ = dmat->row_ptr;
  auto& num_row_ = dmat->num_row;
  auto& num_col_ = dmat->num_col;
  auto& nelem_ = dmat->nelem;

  std::vector<size_t> max_col_ind(nthread, 0);
  parser->BeforeFirst();
  while (parser->Next()) {
    const dmlc::RowBlock<uint32_t>& batch = parser->Value();
    num_row_ += batch.size;
    nelem_ += batch.offset[batch.size];
    const size_t top = data_.size();
    data_.resize(top + batch.offset[batch.size] - batch.offset[0]);
    col_ind_.resize(top + batch.offset[batch.size] - batch.offset[0]);
    CHECK_LT(static_cast<int64_t>(batch.offset[batch.size]),
             std::numeric_limits<int64_t>::max());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (int64_t i = static_cast<int64_t>(batch.offset[0]);
                 i < static_cast<int64_t>(batch.offset[batch.size]); ++i) {
      const int tid = omp_get_thread_num();
      const uint32_t index = batch.index[i];
      const float fvalue = (batch.value == nullptr) ? 1.0f :
                           static_cast<float>(batch.value[i]);
      const size_t offset = top + i - batch.offset[0];
      data_[offset] = fvalue;
      col_ind_[offset] = index;
      max_col_ind[tid] = std::max(max_col_ind[tid],
                                  static_cast<size_t>(index));
    }
    const size_t rtop = row_ptr_.size();
    row_ptr_.resize(rtop + batch.size);
    CHECK_LT(static_cast<int64_t>(batch.size),
             std::numeric_limits<int64_t>::max());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (int64_t i = 0; i < static_cast<int64_t>(batch.size); ++i) {
      row_ptr_[rtop + i]
        = row_ptr_[rtop - 1] + batch.offset[i + 1] - batch.offset[0];
    }
    if (verbose > 0) {
      LOG(INFO) << num_row_ << " rows read into memory";
    }
  }
  num_col_ = *std::max_element(max_col_ind.begin(), max_col_ind.end()) + 1;
  return dmat;
}

template<typename ElementType>
std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(
    std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col) {
  std::unique_ptr<DenseDMatrix> matrix = std::make_unique<DenseDMatrixImpl<ElementType>>(
      std::move(data), missing_value, num_row, num_col
  );
  matrix->type_ = InferTypeInfoOf<ElementType>();
  return matrix;
}

template<typename ElementType>
std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(const void* data, const void* missing_value, size_t num_row, size_t num_col) {
  auto* data_ptr = static_cast<const ElementType*>(data);
  const size_t num_elem = num_row * num_col;
  return DenseDMatrix::Create(std::vector<ElementType>(data_ptr, data_ptr + num_elem),
                              *static_cast<const ElementType*>(missing_value), num_row, num_col);
}

std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(
    TypeInfo type, const void* data, const void* missing_value, size_t num_row, size_t num_col) {
  CHECK(type != TypeInfo::kInvalid) << "ElementType cannot be invalid";
  switch (type) {
  case TypeInfo::kFloat32:
    return Create<float>(data, missing_value, num_row, num_col);
  case TypeInfo::kFloat64:
    return Create<double>(data, missing_value, num_row, num_col);
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
  default:
    LOG(FATAL) << "Invalid type for DenseDMatrix: " << TypeInfoToString(type);
  }
  return std::unique_ptr<DenseDMatrix>(nullptr);
}

template<typename ElementType>
DenseDMatrixImpl<ElementType>::DenseDMatrixImpl(
    std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col)
    : DenseDMatrix(), data(std::move(data)), missing_value(missing_value), num_row(num_row),
      num_col(num_col) {}

template<typename ElementType>
std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(std::vector<ElementType> data, std::vector<uint32_t> col_ind,
                   std::vector<size_t> row_ptr, size_t num_row, size_t num_col) {
  std::unique_ptr<CSRDMatrix> matrix = std::make_unique<CSRDMatrixImpl<ElementType>>(
      std::move(data), std::move(col_ind), std::move(row_ptr), num_row, num_col
  );
  matrix->type_ = InferTypeInfoOf<ElementType>();
  return matrix;
}

template<typename ElementType>
std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(const void* data, const uint32_t* col_ind,
                   const size_t* row_ptr, size_t num_row, size_t num_col, size_t num_elem) {
  auto* data_ptr = static_cast<const ElementType*>(data);
  return CSRDMatrix::Create(
      std::vector<ElementType>(data_ptr, data_ptr + num_elem),
      std::vector<uint32_t>(col_ind, col_ind + num_elem),
      std::vector<size_t>(row_ptr, row_ptr + num_row + 1),
      num_row,
      num_col
  );
}

std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(TypeInfo type, const void* data, const uint32_t* col_ind, const size_t* row_ptr,
                   size_t num_row, size_t num_col, size_t num_elem) {
  CHECK(type != TypeInfo::kInvalid) << "ElementType cannot be invalid";
  switch (type) {
  case TypeInfo::kFloat32:
    return Create<float>(data, col_ind, row_ptr, num_row, num_col, num_elem);
  case TypeInfo::kFloat64:
    return Create<double>(data, col_ind, row_ptr, num_row, num_col, num_elem);
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
  default:
    LOG(FATAL) << "Invalid type for CSRDMatrix: " << TypeInfoToString(type);
  }
  return std::unique_ptr<CSRDMatrix>(nullptr);
}

template <typename ElementType>
CSRDMatrixImpl<ElementType>::CSRDMatrixImpl(
    std::vector<ElementType> data, std::vector<uint32_t> col_ind, std::vector<size_t> row_ptr,
    size_t num_row, size_t num_col)
    : CSRDMatrix(), data(std::move(data)), col_ind(std::move(col_ind)), row_ptr(std::move(row_ptr)),
      num_row(num_col), num_col(num_col)
{}

}  // namespace treelite
