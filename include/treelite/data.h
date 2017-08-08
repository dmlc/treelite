/*!
 * Copyright (c) 2017 by Contributors
 * \file data.h
 * \author Philip Cho
 * \brief Input data structure of treelite
 */
#ifndef TREELITE_DATA_H_
#define TREELITE_DATA_H_

#include <dmlc/data.h>

namespace treelite {

struct DMatrix {
  std::vector<float> data;
  std::vector<size_t> row_ptr;
  std::vector<uint32_t> col_ind;
  size_t num_row;
  size_t num_col;
  size_t nnz;  // number of nonzero entries

  inline void Clear() {
    data.clear();
    row_ptr.clear();
    col_ind.clear();
    row_ptr.resize(1, 0);
    num_row = num_col = nnz = 0;
  }
  static DMatrix* Create(const char* filename, const char* format,
                         int nthread, int verbose);
  static DMatrix* Create(dmlc::Parser<uint32_t>* parser,
                         int nthread, int verbose);
};

}  // namespace treelite

#endif  // TREELITE_DATA_H_
