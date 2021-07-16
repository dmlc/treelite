/*!
 * Copyright (c) 2021 by Contributors
 * \file test_threading_utils.cc
 * \author Hyunsu Cho
 * \brief C++ tests for threading utilities
 */
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cstddef>
#include <cstdint>
#include "threading_utils/parallel_for.h"

using namespace testing;

namespace treelite {
namespace threading_utils {

TEST(ThreadingUtils, ComputeWorkRange) {
  std::random_device rd;
  std::mt19937 rng(rd());
  constexpr int64_t kHigh = 10000;
  std::uniform_int_distribution<int64_t> dist(0, kHigh);
  std::uniform_int_distribution<int64_t> dist2(1, 100);

  /* Test error handling */
  EXPECT_THAT([&]() { ComputeWorkRange(0, 100, 0); },
              ThrowsMessage<treelite::Error>(HasSubstr("nthread must be positive")));
  EXPECT_THAT([&]() { ComputeWorkRange(-100, 100, 3); },
              ThrowsMessage<treelite::Error>(HasSubstr("begin must be 0 or greater")));
  EXPECT_THAT([&]() { ComputeWorkRange(-200, -100, 3); },
              ThrowsMessage<treelite::Error>(HasSubstr("end must be 0 or greater")));
  EXPECT_THAT([&]() { ComputeWorkRange(200, 100, 3); },
              ThrowsMessage<treelite::Error>(HasSubstr("end cannot be less than begin")));

  constexpr int kNumTrial = 1000;
  for (int i = 0; i < kNumTrial; ++i) {
    int64_t begin = dist(rng);
    std::size_t nthread = dist2(rng);
    std::uniform_int_distribution<int64_t> dist3(begin, kHigh);
    int64_t end = dist3(rng);
    auto range = ComputeWorkRange(begin, end, nthread);
    EXPECT_EQ(range.size(), nthread + 1);
    EXPECT_EQ(range[0], begin);
    EXPECT_EQ(range[nthread], end);
    for (std::size_t i = 0; i < nthread; ++i) {
      EXPECT_GE(range[i + 1], range[i]);
    }
  }
}

}  // namespace threading_utils
}  // namespace treelite
