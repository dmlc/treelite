/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file data.cc
 * \author Hyunsu Cho
 * \brief Input data structure of Treelite
 */

#include <treelite/logging.h>
#include <treelite/data.h>
#include <memory>
#include <limits>
#include <cstdint>
#include <cstddef>

namespace treelite {

template<typename ElementType>
std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(
    std::vector<ElementType> data, ElementType missing_value, std::size_t num_row,
    std::size_t num_col) {
  std::unique_ptr<DenseDMatrix> matrix = std::make_unique<DenseDMatrixImpl<ElementType>>(
      std::move(data), missing_value, num_row, num_col);
  matrix->element_type_ = TypeToInfo<ElementType>();
  return matrix;
}

template<typename ElementType>
std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(const void* data, const void* missing_value, std::size_t num_row,
    std::size_t num_col) {
  auto* data_ptr = static_cast<const ElementType*>(data);
  const std::size_t num_elem = num_row * num_col;
  return DenseDMatrix::Create(std::vector<ElementType>(data_ptr, data_ptr + num_elem),
                              *static_cast<const ElementType*>(missing_value), num_row, num_col);
}

std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(
    TypeInfo type, const void* data, const void* missing_value, std::size_t num_row,
    std::size_t num_col) {
  TREELITE_CHECK(type != TypeInfo::kInvalid) << "ElementType cannot be invalid";
  switch (type) {
  case TypeInfo::kFloat32:
    return Create<float>(data, missing_value, num_row, num_col);
  case TypeInfo::kFloat64:
    return Create<double>(data, missing_value, num_row, num_col);
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
  default:
    TREELITE_LOG(FATAL) << "Invalid type for DenseDMatrix: " << TypeInfoToString(type);
  }
  return std::unique_ptr<DenseDMatrix>(nullptr);
}

TypeInfo
DenseDMatrix::GetElementType() const {
  return element_type_;
}

template<typename ElementType>
DenseDMatrixImpl<ElementType>::DenseDMatrixImpl(
    std::vector<ElementType> data, ElementType missing_value, std::size_t num_row,
    std::size_t num_col)
    : DenseDMatrix(), data(std::move(data)), missing_value(missing_value), num_row(num_row),
      num_col(num_col) {}

template<typename ElementType>
std::size_t
DenseDMatrixImpl<ElementType>::GetNumRow() const {
  return num_row;
}

template<typename ElementType>
std::size_t
DenseDMatrixImpl<ElementType>::GetNumCol() const {
  return num_col;
}

template<typename ElementType>
std::size_t
DenseDMatrixImpl<ElementType>::GetNumElem() const {
  return num_row * num_col;
}

template<typename ElementType>
DMatrixType
DenseDMatrixImpl<ElementType>::GetType() const {
  return DMatrixType::kDense;
}

template <typename ElementType>
template <typename OutputType>
void
DenseDMatrixImpl<ElementType>::FillRow(std::size_t row_id, OutputType* out) const {
  std::size_t out_idx = 0;
  std::size_t in_idx = row_id * num_col;
  while (out_idx < num_col) {
    out[out_idx] = static_cast<OutputType>(data[in_idx]);
    ++out_idx;
    ++in_idx;
  }
}

template <typename ElementType>
template <typename OutputType>
void
DenseDMatrixImpl<ElementType>::ClearRow(std::size_t row_id, OutputType* out) const {
  for (std::size_t i = 0; i < num_col; ++i) {
    out[i] = std::numeric_limits<OutputType>::quiet_NaN();
  }
}

template<typename ElementType>
std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(std::vector<ElementType> data, std::vector<std::uint32_t> col_ind,
                   std::vector<std::size_t> row_ptr, std::size_t num_row, std::size_t num_col) {
  std::unique_ptr<CSRDMatrix> matrix = std::make_unique<CSRDMatrixImpl<ElementType>>(
      std::move(data), std::move(col_ind), std::move(row_ptr), num_row, num_col);
  matrix->element_type_ = TypeToInfo<ElementType>();
  return matrix;
}

template<typename ElementType>
std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(const void* data, const std::uint32_t* col_ind,
                   const std::size_t* row_ptr, std::size_t num_row, std::size_t num_col) {
  auto* data_ptr = static_cast<const ElementType*>(data);
  const std::size_t num_elem = row_ptr[num_row];
  return CSRDMatrix::Create(
      std::vector<ElementType>(data_ptr, data_ptr + num_elem),
      std::vector<std::uint32_t>(col_ind, col_ind + num_elem),
      std::vector<std::size_t>(row_ptr, row_ptr + num_row + 1),
      num_row,
      num_col);
}

std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(TypeInfo type, const void* data, const std::uint32_t* col_ind,
                   const std::size_t* row_ptr, std::size_t num_row, std::size_t num_col) {
  TREELITE_CHECK(type != TypeInfo::kInvalid) << "ElementType cannot be invalid";
  switch (type) {
  case TypeInfo::kFloat32:
    return Create<float>(data, col_ind, row_ptr, num_row, num_col);
  case TypeInfo::kFloat64:
    return Create<double>(data, col_ind, row_ptr, num_row, num_col);
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
  default:
    TREELITE_LOG(FATAL) << "Invalid type for CSRDMatrix: " << TypeInfoToString(type);
  }
  return std::unique_ptr<CSRDMatrix>(nullptr);
}

TypeInfo
CSRDMatrix::GetElementType() const {
  return element_type_;
}

template <typename ElementType>
CSRDMatrixImpl<ElementType>::CSRDMatrixImpl(
    std::vector<ElementType> data, std::vector<std::uint32_t> col_ind,
    std::vector<std::size_t> row_ptr, std::size_t num_row, std::size_t num_col)
    : CSRDMatrix(), data(std::move(data)), col_ind(std::move(col_ind)), row_ptr(std::move(row_ptr)),
      num_row(num_row), num_col(num_col)
{}

template <typename ElementType>
std::size_t
CSRDMatrixImpl<ElementType>::GetNumRow() const {
  return num_row;
}

template <typename ElementType>
std::size_t
CSRDMatrixImpl<ElementType>::GetNumCol() const {
  return num_col;
}

template <typename ElementType>
std::size_t
CSRDMatrixImpl<ElementType>::GetNumElem() const {
  return row_ptr.at(num_row);
}

template <typename ElementType>
DMatrixType
CSRDMatrixImpl<ElementType>::GetType() const {
  return DMatrixType::kSparseCSR;
}

template <typename ElementType>
template <typename OutputType>
void
CSRDMatrixImpl<ElementType>::FillRow(std::size_t row_id, OutputType* out) const {
  for (std::size_t i = row_ptr[row_id]; i < row_ptr[row_id + 1]; ++i) {
    out[col_ind[i]] = static_cast<OutputType>(data[i]);
  }
}

template <typename ElementType>
template <typename OutputType>
void
CSRDMatrixImpl<ElementType>::ClearRow(std::size_t row_id, OutputType* out) const {
  for (std::size_t i = row_ptr[row_id]; i < row_ptr[row_id + 1]; ++i) {
    out[col_ind[i]] = std::numeric_limits<OutputType>::quiet_NaN();
  }
}

template class DenseDMatrixImpl<float>;
template class DenseDMatrixImpl<double>;
template class CSRDMatrixImpl<float>;
template class CSRDMatrixImpl<double>;

template void CSRDMatrixImpl<float>::FillRow<float>(std::size_t, float*) const;
template void CSRDMatrixImpl<float>::FillRow<double>(std::size_t, double*) const;
template void CSRDMatrixImpl<float>::ClearRow<float>(std::size_t, float*) const;
template void CSRDMatrixImpl<float>::ClearRow<double>(std::size_t, double*) const;
template void CSRDMatrixImpl<double>::FillRow<float>(std::size_t, float*) const;
template void CSRDMatrixImpl<double>::FillRow<double>(std::size_t, double*) const;
template void CSRDMatrixImpl<double>::ClearRow<float>(std::size_t, float*) const;
template void CSRDMatrixImpl<double>::ClearRow<double>(std::size_t, double*) const;
template void DenseDMatrixImpl<float>::FillRow<float>(std::size_t, float*) const;
template void DenseDMatrixImpl<float>::FillRow<double>(std::size_t, double*) const;
template void DenseDMatrixImpl<float>::ClearRow<float>(std::size_t, float*) const;
template void DenseDMatrixImpl<float>::ClearRow<double>(std::size_t, double*) const;
template void DenseDMatrixImpl<double>::FillRow<float>(std::size_t, float*) const;
template void DenseDMatrixImpl<double>::FillRow<double>(std::size_t, double*) const;
template void DenseDMatrixImpl<double>::ClearRow<float>(std::size_t, float*) const;
template void DenseDMatrixImpl<double>::ClearRow<double>(std::size_t, double*) const;

}  // namespace treelite
