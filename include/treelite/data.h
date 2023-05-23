/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file data.h
 * \author Hyunsu Cho
 * \brief Input data structure of Treelite
 */
#ifndef TREELITE_DATA_H_
#define TREELITE_DATA_H_

#include <treelite/typeinfo.h>

#include <memory>
#include <type_traits>
#include <vector>

namespace treelite {

enum class DMatrixType : uint8_t { kDense = 0, kSparseCSR = 1 };

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
  template <typename ElementType>
  static std::unique_ptr<DenseDMatrix> Create(
      std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col);
  template <typename ElementType>
  static std::unique_ptr<DenseDMatrix> Create(
      void const* data, void const* missing_value, size_t num_row, size_t num_col);
  static std::unique_ptr<DenseDMatrix> Create(
      TypeInfo type, void const* data, void const* missing_value, size_t num_row, size_t num_col);
  size_t GetNumRow() const override = 0;
  size_t GetNumCol() const override = 0;
  size_t GetNumElem() const override = 0;
  DMatrixType GetType() const override = 0;
  TypeInfo GetElementType() const override;
};

template <typename ElementType>
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
  DenseDMatrixImpl(
      std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col);
  ~DenseDMatrixImpl() = default;
  DenseDMatrixImpl(DenseDMatrixImpl const&) = default;
  DenseDMatrixImpl(DenseDMatrixImpl&&) noexcept = default;
  DenseDMatrixImpl& operator=(DenseDMatrixImpl const&) = default;
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
  template <typename ElementType>
  static std::unique_ptr<CSRDMatrix> Create(std::vector<ElementType> data,
      std::vector<uint32_t> col_ind, std::vector<size_t> row_ptr, size_t num_row, size_t num_col);
  template <typename ElementType>
  static std::unique_ptr<CSRDMatrix> Create(void const* data, uint32_t const* col_ind,
      size_t const* row_ptr, size_t num_row, size_t num_col);
  static std::unique_ptr<CSRDMatrix> Create(TypeInfo type, void const* data,
      uint32_t const* col_ind, size_t const* row_ptr, size_t num_row, size_t num_col);
  size_t GetNumRow() const override = 0;
  size_t GetNumCol() const override = 0;
  size_t GetNumElem() const override = 0;
  DMatrixType GetType() const override = 0;
  TypeInfo GetElementType() const override;
};

template <typename ElementType>
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
  CSRDMatrixImpl(CSRDMatrixImpl const&) = default;
  CSRDMatrixImpl(CSRDMatrixImpl&&) noexcept = default;
  CSRDMatrixImpl& operator=(CSRDMatrixImpl const&) = default;
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
