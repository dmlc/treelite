#ifndef TREELITE_COMPILER_COMMON_CATEGORICAL_BITMAP_H_
#define TREELITE_COMPILER_COMMON_CATEGORICAL_BITMAP_H_

#include <vector>

namespace treelite {
namespace compiler {
namespace common_util {

inline std::vector<uint64_t>
GetCategoricalBitmap(const std::vector<uint32_t>& left_categories) {
  const size_t num_left_categories = left_categories.size();
  const uint32_t max_left_category = left_categories[num_left_categories - 1];
  std::vector<uint64_t> bitmap((max_left_category + 1 + 63) / 64, 0);
  for (size_t i = 0; i < left_categories.size(); ++i) {
    const uint32_t cat = left_categories[i];
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
