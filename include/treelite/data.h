/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file data.h
 * \author Hyunsu Cho
 * \brief Input data structure of Treelite
 */
#ifndef TREELITE_DATA_H_
#define TREELITE_DATA_H_

#include <dmlc/data.h>
#include <treelite/typeinfo.h>
#include <vector>
#include <type_traits>
#include <memory>

namespace treelite {

class DMatrix {
 public:
  virtual size_t GetNumRow() const = 0;
  virtual size_t GetNumCol() const = 0;
  virtual size_t GetNumElem() const = 0;
  DMatrix() = default;
  virtual ~DMatrix() = default;
};

class DenseDMatrix : public DMatrix {
 private:
  TypeInfo element_type_;
 public:
  template<typename ElementType>
  static std::unique_ptr<DenseDMatrix> Create(
      std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col);
  template<typename ElementType>
  static std::unique_ptr<DenseDMatrix> Create(
      const void* data, const void* missing_value, size_t num_row, size_t num_col);
  static std::unique_ptr<DenseDMatrix> Create(
      TypeInfo type, const void* data, const void* missing_value, size_t num_row, size_t num_col);
  size_t GetNumRow() const override = 0;
  size_t GetNumCol() const override = 0;
  size_t GetNumElem() const override = 0;
};

template<typename ElementType>
class DenseDMatrixImpl : public DenseDMatrix {
 private:
  /*! \brief feature values */
  std::vector<ElementType> data;
  /*! \brief value representing the missing value (usually NaN) */
  ElementType missing_value;
  /*! \brief number of rows */
  size_t num_row;
  /*! \brief number of columns (i.e. # of features used) */
  size_t num_col;
 public:
  DenseDMatrixImpl() = delete;
  DenseDMatrixImpl(std::vector<ElementType> data, ElementType missing_value, size_t num_row,
                   size_t num_col);
  ~DenseDMatrixImpl() = default;
  DenseDMatrixImpl(const DenseDMatrixImpl&) = default;
  DenseDMatrixImpl(DenseDMatrixImpl&&) noexcept = default;
  DenseDMatrixImpl& operator=(const DenseDMatrixImpl&) = default;
  DenseDMatrixImpl& operator=(DenseDMatrixImpl&&) noexcept = default;

  size_t GetNumRow() const override;
  size_t GetNumCol() const override;
  size_t GetNumElem() const override;

  friend class DenseDMatrix;
  static_assert(std::is_same<ElementType, float>::value || std::is_same<ElementType, double>::value,
                "ElementType must be either float32 or float64");
};

class CSRDMatrix : public DMatrix {
 private:
  TypeInfo element_type_;
 public:
  template<typename ElementType>
  static std::unique_ptr<CSRDMatrix> Create(
      std::vector<ElementType> data, std::vector<uint32_t> col_ind, std::vector<size_t> row_ptr,
      size_t num_row, size_t num_col);
  template<typename ElementType>
  static std::unique_ptr<CSRDMatrix> Create(
      const void* data, const uint32_t* col_ind, const size_t* row_ptr, size_t num_row,
      size_t num_col);
  static std::unique_ptr<CSRDMatrix> Create(
      TypeInfo type, const void* data, const uint32_t* col_ind, const size_t* row_ptr,
      size_t num_row, size_t num_col);
  static std::unique_ptr<CSRDMatrix> Create(
      const char* filename, const char* format, int nthread, int verbose);
  TypeInfo GetMatrixType() const;
  template <typename Func>
  inline auto Dispatch(Func func) const;
  size_t GetNumRow() const override = 0;
  size_t GetNumCol() const override = 0;
  size_t GetNumElem() const override = 0;
};

template<typename ElementType>
class CSRDMatrixImpl : public CSRDMatrix {
 public:
  /*! \brief feature values */
  std::vector<ElementType> data;
  /*! \brief feature indices. col_ind[i] indicates the feature index associated with data[i]. */
  std::vector<uint32_t> col_ind;
  /*! \brief pointer to row headers; length is [num_row] + 1. */
  std::vector<size_t> row_ptr;
  /*! \brief number of rows */
  size_t num_row;
  /*! \brief number of columns (i.e. # of features used) */
  size_t num_col;

  CSRDMatrixImpl() = delete;
  CSRDMatrixImpl(std::vector<ElementType> data, std::vector<uint32_t> col_ind,
                 std::vector<size_t> row_ptr, size_t num_row, size_t num_col);
  ~CSRDMatrixImpl() = default;
  CSRDMatrixImpl(const CSRDMatrixImpl&) = default;
  CSRDMatrixImpl(CSRDMatrixImpl&&) noexcept = default;
  CSRDMatrixImpl& operator=(const CSRDMatrixImpl&) = default;
  CSRDMatrixImpl& operator=(CSRDMatrixImpl&&) noexcept = default;

  size_t GetNumRow() const override;
  size_t GetNumCol() const override;
  size_t GetNumElem() const override;

  friend class CSRDMatrix;
  static_assert(std::is_same<ElementType, float>::value || std::is_same<ElementType, double>::value,
                "ElementType must be either float32 or float64");
};

template <typename Func>
inline auto
CSRDMatrix::Dispatch(Func func) const {
  switch (element_type_) {
  case TypeInfo::kFloat32:
    return func(*dynamic_cast<const CSRDMatrixImpl<float>*>(this));
    break;
  case TypeInfo::kFloat64:
    return func(*dynamic_cast<const CSRDMatrixImpl<double>*>(this));
    break;
  case TypeInfo::kUInt32:
  case TypeInfo::kInvalid:
  default:
    LOG(FATAL) << "Invalid element type for the matrix: " << TypeInfoToString(element_type_);
    return func(*dynamic_cast<const CSRDMatrixImpl<double>*>(this));  // avoid missing return error
  }
}

}  // namespace treelite

#endif  // TREELITE_DATA_H_
