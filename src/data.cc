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

namespace {

template <typename ElementType, typename DMLCParserDType>
inline static std::unique_ptr<treelite::CSRDMatrix> CreateFromParserImpl(
    const char* filename, const char* format, int nthread, int verbose) {
  std::unique_ptr<dmlc::Parser<uint32_t, DMLCParserDType>> parser(
      dmlc::Parser<uint32_t, DMLCParserDType>::Create(filename, 0, 1, format));

  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);

  std::vector<ElementType> data;
  std::vector<uint32_t> col_ind;
  std::vector<size_t> row_ptr;
  row_ptr.resize(1, 0);
  size_t num_row = 0;
  size_t num_col = 0;
  size_t num_elem = 0;

  std::vector<size_t> max_col_ind(nthread, 0);
  parser->BeforeFirst();
  while (parser->Next()) {
    const dmlc::RowBlock<uint32_t, DMLCParserDType>& batch = parser->Value();
    num_row += batch.size;
    num_elem += batch.offset[batch.size];
    const size_t top = data.size();
    data.resize(top + batch.offset[batch.size] - batch.offset[0]);
    col_ind.resize(top + batch.offset[batch.size] - batch.offset[0]);
    CHECK_LT(static_cast<int64_t>(batch.offset[batch.size]),
             std::numeric_limits<int64_t>::max());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (int64_t i = static_cast<int64_t>(batch.offset[0]);
         i < static_cast<int64_t>(batch.offset[batch.size]); ++i) {
      const int tid = omp_get_thread_num();
      const uint32_t index = batch.index[i];
      const ElementType fvalue
        = ((batch.value == nullptr) ? static_cast<ElementType>(1)
                                    : static_cast<ElementType>(batch.value[i]));
      const size_t offset = top + i - batch.offset[0];
      data[offset] = fvalue;
      col_ind[offset] = index;
      max_col_ind[tid] = std::max(max_col_ind[tid], static_cast<size_t>(index));
    }
    const size_t rtop = row_ptr.size();
    row_ptr.resize(rtop + batch.size);
    CHECK_LT(static_cast<int64_t>(batch.size), std::numeric_limits<int64_t>::max());
    #pragma omp parallel for schedule(static) num_threads(nthread)
    for (int64_t i = 0; i < static_cast<int64_t>(batch.size); ++i) {
      row_ptr[rtop + i] = row_ptr[rtop - 1] + batch.offset[i + 1] - batch.offset[0];
    }
    if (verbose > 0) {
      LOG(INFO) << num_row << " rows read into memory";
    }
  }
  num_col = *std::max_element(max_col_ind.begin(), max_col_ind.end()) + 1;
  return treelite::CSRDMatrix::Create(std::move(data), std::move(col_ind), std::move(row_ptr),
                                      num_row, num_col);
}

std::unique_ptr<treelite::CSRDMatrix>
CreateFromParser(
    const char* filename, const char* format, treelite::TypeInfo dtype, int nthread, int verbose) {
  switch (dtype) {
  case treelite::TypeInfo::kFloat32:
    return CreateFromParserImpl<float, float>(filename, format, nthread, verbose);
  case treelite::TypeInfo::kFloat64:
    return CreateFromParserImpl<double, float>(filename, format, nthread, verbose);
  case treelite::TypeInfo::kUInt32:
    return CreateFromParserImpl<uint32_t, int64_t>(filename, format, nthread, verbose);
  default:
    LOG(FATAL) << "Unrecognized TypeInfo: " << treelite::TypeInfoToString(dtype);
  }
  return CreateFromParserImpl<float, float>(filename, format, nthread, verbose);
    // avoid missing value warning
}

}  // anonymous namespace

namespace treelite {

template<typename ElementType>
std::unique_ptr<DenseDMatrix>
DenseDMatrix::Create(
    std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col) {
  std::unique_ptr<DenseDMatrix> matrix = std::make_unique<DenseDMatrixImpl<ElementType>>(
      std::move(data), missing_value, num_row, num_col);
  matrix->element_type_ = TypeToInfo<ElementType>();
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

TypeInfo
DenseDMatrix::GetElementType() const {
  return element_type_;
}

template<typename ElementType>
DenseDMatrixImpl<ElementType>::DenseDMatrixImpl(
    std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col)
    : DenseDMatrix(), data(std::move(data)), missing_value(missing_value), num_row(num_row),
      num_col(num_col) {}

template<typename ElementType>
size_t
DenseDMatrixImpl<ElementType>::GetNumRow() const {
  return num_row;
}

template<typename ElementType>
size_t
DenseDMatrixImpl<ElementType>::GetNumCol() const {
  return num_col;
}

template<typename ElementType>
size_t
DenseDMatrixImpl<ElementType>::GetNumElem() const {
  return num_row * num_col;
}

template<typename ElementType>
DMatrixType
DenseDMatrixImpl<ElementType>::GetType() const {
  return DMatrixType::kDense;
}

template<typename ElementType>
std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(std::vector<ElementType> data, std::vector<uint32_t> col_ind,
                   std::vector<size_t> row_ptr, size_t num_row, size_t num_col) {
  std::unique_ptr<CSRDMatrix> matrix = std::make_unique<CSRDMatrixImpl<ElementType>>(
      std::move(data), std::move(col_ind), std::move(row_ptr), num_row, num_col);
  matrix->element_type_ = TypeToInfo<ElementType>();
  return matrix;
}

template<typename ElementType>
std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(const void* data, const uint32_t* col_ind,
                   const size_t* row_ptr, size_t num_row, size_t num_col) {
  auto* data_ptr = static_cast<const ElementType*>(data);
  const size_t num_elem = row_ptr[num_row];
  return CSRDMatrix::Create(
      std::vector<ElementType>(data_ptr, data_ptr + num_elem),
      std::vector<uint32_t>(col_ind, col_ind + num_elem),
      std::vector<size_t>(row_ptr, row_ptr + num_row + 1),
      num_row,
      num_col);
}

std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(TypeInfo type, const void* data, const uint32_t* col_ind, const size_t* row_ptr,
                   size_t num_row, size_t num_col) {
  CHECK(type != TypeInfo::kInvalid) << "ElementType cannot be invalid";
  switch (type) {
  case TypeInfo::kFloat32:
    return Create<float>(data, col_ind, row_ptr, num_row, num_col);
  case TypeInfo::kFloat64:
    return Create<double>(data, col_ind, row_ptr, num_row, num_col);
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
  default:
    LOG(FATAL) << "Invalid type for CSRDMatrix: " << TypeInfoToString(type);
  }
  return std::unique_ptr<CSRDMatrix>(nullptr);
}

std::unique_ptr<CSRDMatrix>
CSRDMatrix::Create(
    const char* filename, const char* format, const char* data_type, int nthread, int verbose) {
  TypeInfo dtype = (data_type ? GetTypeInfoByName(data_type) : TypeInfo::kFloat32);
  return CreateFromParser(filename, format, dtype, nthread, verbose);
}

TypeInfo
CSRDMatrix::GetElementType() const {
  return element_type_;
}

template <typename ElementType>
CSRDMatrixImpl<ElementType>::CSRDMatrixImpl(
    std::vector<ElementType> data, std::vector<uint32_t> col_ind, std::vector<size_t> row_ptr,
    size_t num_row, size_t num_col)
    : CSRDMatrix(), data(std::move(data)), col_ind(std::move(col_ind)), row_ptr(std::move(row_ptr)),
      num_row(num_row), num_col(num_col)
{}

template <typename ElementType>
size_t
CSRDMatrixImpl<ElementType>::GetNumRow() const {
  return num_row;
}

template <typename ElementType>
size_t
CSRDMatrixImpl<ElementType>::GetNumCol() const {
  return num_col;
}

template <typename ElementType>
size_t
CSRDMatrixImpl<ElementType>::GetNumElem() const {
  return row_ptr.at(num_row);
}

template <typename ElementType>
DMatrixType
CSRDMatrixImpl<ElementType>::GetType() const {
  return DMatrixType::kSparseCSR;
}

template class DenseDMatrixImpl<float>;
template class DenseDMatrixImpl<double>;
template class CSRDMatrixImpl<float>;
template class CSRDMatrixImpl<double>;

}  // namespace treelite
