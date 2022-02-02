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

namespace {

class RandomGenerator {
 public:
  RandomGenerator()
    : rng_(std::random_device()()),
      int_dist_(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()),
      real_dist_(0.0, 1.0) {}

  int64_t DrawInteger(int64_t low, int64_t high) {
    TREELITE_CHECK_LT(low, high);
    int64_t out = int_dist_(rng_);
    int64_t rem = out % (high - low);
    int64_t ret;
    if (rem < 0) {
      ret = high + rem;
    } else {
      ret = low + rem;
    }
    TREELITE_CHECK_GE(ret, low);
    TREELITE_CHECK_LT(ret, high);
    return ret;
  }

  double DrawReal(double low, double high) {
    TREELITE_CHECK_LT(low, high);
    return real_dist_(rng_) * (high - low) + low;
  }

 private:
  std::mt19937 rng_;
  std::uniform_int_distribution<int64_t> int_dist_;
  std::uniform_real_distribution<double> real_dist_;
};

}  // namespace anonymous

namespace treelite {
namespace threading_utils {
TEST(ThreadingUtils, ParallelFor) {
  /* Test error handling */
  const int max_thread = treelite::threading_utils::MaxNumThread();

  auto sched = treelite::threading_utils::ParallelSchedule::Guided();

  auto dummy_func = [](int, std::size_t) {};
  EXPECT_THROW(ParallelFor(0, 100, 0, sched, dummy_func), treelite::Error);
  EXPECT_THROW(ParallelFor(200, 100, 3, sched, dummy_func), treelite::Error);
  EXPECT_THROW(ParallelFor(-100, 100, 3, sched, dummy_func), treelite::Error);
  EXPECT_THROW(ParallelFor(-200, -100, 3, sched, dummy_func), treelite::Error);
  EXPECT_THROW(ParallelFor(200, 100, 3, sched, dummy_func), treelite::Error);
  EXPECT_THROW(ParallelFor(10, 20, 3 * max_thread, sched, dummy_func), treelite::Error);

  /* Property-based testing with randomly generated parameters */
  constexpr int kVectorLength = 10000;
  RandomGenerator rng;
  std::vector<double> a(kVectorLength);
  std::vector<double> b(kVectorLength);
  std::generate_n(a.begin(), kVectorLength, [&rng]() { return rng.DrawReal(-1.0, 1.0); });
  std::generate_n(b.begin(), kVectorLength, [&rng]() { return rng.DrawReal(-10.0, 10.0); });

  constexpr int kNumTrial = 200;
  for (int i = 0; i < kNumTrial; ++i) {
    std::vector<double> c(kVectorLength);
    // Fill c with dummy values
    std::generate_n(c.begin(), kVectorLength, [&rng]() { return rng.DrawReal(100.0, 200.0); });

    // Compute c := a + b on range [begin, end)
    int64_t begin = rng.DrawInteger(0, kVectorLength);
    std::size_t nthread = static_cast<std::size_t>(rng.DrawInteger(1, max_thread + 1));
    int64_t end = rng.DrawInteger(begin, kVectorLength);

    ParallelFor(begin, end, nthread, sched, [&a, &b, &c](int64_t i, std::size_t) {
      c[i] = a[i] + b[i];
    });

    for (int64_t i = begin; i < end; ++i) {
      EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << ", at index " << i;
    }
  }
}

}  // namespace threading_utils
}  // namespace treelite
