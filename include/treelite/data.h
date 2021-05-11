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

enum class DMatrixType : uint8_t {
  kDense = 0,
  kSparseCSR = 1
};

class DMatrix {
 public:
  virtual size_t GetNumRow() const = 0;
  virtual size_t GetNumCol() const = 0;
  virtual size_t GetNumElem() const = 0;
  virtual DMatrixType GetType() const = 0;
  virtual TypeInfo GetElementType() const = 0;
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
  DMatrixType GetType() const override = 0;
  TypeInfo GetElementType() const override;
};

template<typename ElementType>
class DenseDMatrixImpl : public DenseDMatrix {
 public:
  /*! \brief feature values */
  std::vector<ElementType> data;
  /*! \brief value representing the missing value (usually NaN) */
  ElementType missing_value;
  /*! \brief number of rows */
  size_t num_row;
  /*! \brief number of columns (i.e. # of features used) */
  size_t num_col;

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
  DMatrixType GetType() const override;

  template <typename OutputType>
  void FillRow(size_t row_id, OutputType* out) const;
  template <typename OutputType>
  void ClearRow(size_t row_id, OutputType* out) const;

  friend class DenseDMatrix;
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
      const char* filename, const char* format, const char* data_type, int nthread, int verbose);
  size_t GetNumRow() const override = 0;
  size_t GetNumCol() const override = 0;
  size_t GetNumElem() const override = 0;
  DMatrixType GetType() const override = 0;
  TypeInfo GetElementType() const override;
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
  CSRDMatrixImpl(const CSRDMatrixImpl&) = default;
  CSRDMatrixImpl(CSRDMatrixImpl&&) noexcept = default;
  CSRDMatrixImpl& operator=(const CSRDMatrixImpl&) = default;
  CSRDMatrixImpl& operator=(CSRDMatrixImpl&&) noexcept = default;

  size_t GetNumRow() const override;
  size_t GetNumCol() const override;
  size_t GetNumElem() const override;
  DMatrixType GetType() const override;

  template <typename OutputType>
  void FillRow(size_t row_id, OutputType* out) const;
  template <typename OutputType>
  void ClearRow(size_t row_id, OutputType* out) const;

  friend class CSRDMatrix;
};

}  // namespace treelite

#endif  // TREELITE_DATA_H_
