/*!
 * Copyright (c) 2017 by Contributors
 * \file data.h
 * \author Hyunsu Cho
 * \brief Input data structure of treelite
 */
#ifndef TREELITE_DATA_H_
#define TREELITE_DATA_H_

#include <dmlc/data.h>
#include <vector>

namespace treelite {

/*! \brief a simple data matrix in CSR (Compressed Sparse Row) storage */
struct DMatrix {
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
  static DMatrix* Create(const char* filename, const char* format,
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
  static DMatrix* Create(dmlc::Parser<uint32_t>* parser,
                         int nthread, int verbose);
};

}  // namespace treelite

#endif  // TREELITE_DATA_H_
