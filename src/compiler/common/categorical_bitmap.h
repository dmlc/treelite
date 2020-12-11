/*!
 * Copyright (c) 2018-2020 by Contributors
 * \file categorical_bitmap.h
 * \author Hyunsu Cho
 * \brief Function to generate bitmaps for categorical splits
 */
#ifndef TREELITE_COMPILER_COMMON_CATEGORICAL_BITMAP_H_
#define TREELITE_COMPILER_COMMON_CATEGORICAL_BITMAP_H_

#include <vector>

namespace treelite {
namespace compiler {
namespace common_util {

inline std::vector<uint64_t>
GetCategoricalBitmap(const std::vector<uint32_t>& matching_categories) {
  const size_t num_matching_categories = matching_categories.size();
  if (num_matching_categories == 0) {
    return std::vector<uint64_t>{0};
  }
  const uint32_t max_matching_category = matching_categories[num_matching_categories - 1];
  std::vector<uint64_t> bitmap((max_matching_category + 1 + 63) / 64, 0);
  for (uint32_t cat : matching_categories) {
    const size_t idx = cat / 64;
    const uint32_t offset = cat % 64;
    bitmap[idx] |= (static_cast<uint64_t>(1) << offset);
  }
  return bitmap;
}

}  // namespace common_util
}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_COMMON_CATEGORICAL_BITMAP_H_
