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

/*! \brief a simple data matrix in CSR (Compressed Sparse Row) storage */
struct LegacyDMatrix {
  /*! \brief feature values */
  std::vector<float> data;
  /*! \brief feature indices */
  std::vector<uint32_t> col_ind;
  /*! \brief pointer to row headers; length of [num_row] + 1 */
  std::vector<size_t> row_ptr;
  /*! \brief number of rows */
  size_t num_row;
  /*! \brief number of columns */
  size_t num_col;
  /*! \brief number of nonzero entries */
  size_t nelem;

  /*!
   * \brief clear all data fields
   */
  inline void Clear() {
    data.clear();
    row_ptr.clear();
    col_ind.clear();
    row_ptr.resize(1, 0);
    num_row = num_col = nelem = 0;
  }
  /*!
   * \brief construct a new DMatrix from a file
   * \param filename name of file
   * \param format format of file (libsvm/libfm/csv)
   * \param nthread number of threads to use
   * \param verbose whether to produce extra messages
   * \return newly built DMatrix
   */
  static LegacyDMatrix* Create(const char* filename, const char* format,
                               int nthread, int verbose);
  /*!
   * \brief construct a new DMatrix from a data parser. The data parser here
   *        refers to any iterable object that streams input data in small
   *        batches.
   * \param parser pointer to data parser
   * \param nthread number of threads to use
   * \param verbose whether to produce extra messages
   * \return newly built DMatrix
   */
  static LegacyDMatrix* Create(dmlc::Parser<uint32_t>* parser,
                               int nthread, int verbose);
};

class DenseDMatrix {
 private:
  TypeInfo type_;
 public:
  template<typename ElementType>
  static std::unique_ptr<DenseDMatrix> Create(
      std::vector<ElementType> data, ElementType missing_value, size_t num_row, size_t num_col);
  template<typename ElementType>
  static std::unique_ptr<DenseDMatrix> Create(
      const void* data, const void* missing_value, size_t num_row, size_t num_col);
  static std::unique_ptr<DenseDMatrix> Create(
      TypeInfo type, const void* data, const void* missing_value, size_t num_row, size_t num_col);
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

  friend class DenseDMatrix;
  static_assert(std::is_same<ElementType, float>::value || std::is_same<ElementType, double>::value,
                "ElementType must be either float32 or float64");
};

class CSRDMatrix {
 private:
  TypeInfo type_;
 public:
  template<typename ElementType>
  static std::unique_ptr<CSRDMatrix> Create(
      std::vector<ElementType> data, std::vector<uint32_t> col_ind, std::vector<size_t> row_ptr,
      size_t num_row, size_t num_col);
  template<typename ElementType>
  static std::unique_ptr<CSRDMatrix> Create(
      const void* data, const uint32_t* col_ind, const size_t* row_ptr, size_t num_row,
      size_t num_col, size_t num_elem);
  static std::unique_ptr<CSRDMatrix> Create(
      TypeInfo type, const void* data, const uint32_t* col_ind, const size_t* row_ptr,
      size_t num_row, size_t num_col, size_t num_elem);
};

template<typename ElementType>
class CSRDMatrixImpl : public CSRDMatrix {
 private:
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

 public:
  CSRDMatrixImpl() = delete;
  CSRDMatrixImpl(std::vector<ElementType> data, std::vector<uint32_t> col_ind,
                 std::vector<size_t> row_ptr, size_t num_row, size_t num_col);
  ~CSRDMatrixImpl() = default;
  CSRDMatrixImpl(const CSRDMatrixImpl&) = default;
  CSRDMatrixImpl(CSRDMatrixImpl&&) noexcept = default;

  friend class CSRDMatrix;
  static_assert(std::is_same<ElementType, float>::value || std::is_same<ElementType, double>::value,
                "ElementType must be either float32 or float64");
};

}  // namespace treelite

#endif  // TREELITE_DATA_H_
