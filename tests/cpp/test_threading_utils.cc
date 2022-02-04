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
      int_dist_(std::numeric_limits<std::int64_t>::min(),
                std::numeric_limits<std::int64_t>::max()),
      real_dist_(0.0, 1.0) {}

  std::int64_t DrawInteger(std::int64_t low, std::int64_t high) {
    TREELITE_CHECK_LT(low, high);
    std::int64_t out = int_dist_(rng_);
    std::int64_t rem = out % (high - low);
    std::int64_t ret;
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
  std::uniform_int_distribution<std::int64_t> int_dist_;
  std::uniform_real_distribution<double> real_dist_;
};

}  // namespace anonymous

namespace treelite {
namespace threading_utils {

class ThreadingUtilsTestFixture : public ::testing::TestWithParam<ParallelSchedule> {};

TEST_P(ThreadingUtilsTestFixture, ParallelFor) {
  auto sched = GetParam();

  /* Test error handling */
  const int max_thread = treelite::threading_utils::MaxNumThread();
  EXPECT_THROW(threading_utils::ConfigureThreadConfig(max_thread * 3), treelite::Error);

  /* Property-based testing with randomly generated parameters */
  constexpr int kVectorLength = 10000;
  RandomGenerator rng;
  std::vector<double> a(kVectorLength);
  std::vector<double> b(kVectorLength);
  std::generate_n(a.begin(), kVectorLength, [&rng]() { return rng.DrawReal(-1.0, 1.0); });
  std::generate_n(b.begin(), kVectorLength, [&rng]() { return rng.DrawReal(-10.0, 10.0); });

  constexpr int kNumTrial = 200;
  for (int trial_id = 0; trial_id < kNumTrial; ++trial_id) {
    std::vector<double> c(kVectorLength);
    // Fill c with dummy values
    std::generate_n(c.begin(), kVectorLength, [&rng]() { return rng.DrawReal(100.0, 200.0); });

    // Compute c := a + b on range [begin, end)
    std::int64_t begin = rng.DrawInteger(0, kVectorLength);
    auto thread_config = threading_utils::ConfigureThreadConfig(
        static_cast<int>(rng.DrawInteger(1, max_thread + 1)));
    std::int64_t end = rng.DrawInteger(begin, kVectorLength);

    ParallelFor(begin, end, thread_config, sched, [&a, &b, &c](std::int64_t i, int) {
      c[i] = a[i] + b[i];
    });

    for (std::int64_t k = begin; k < end; ++k) {
      EXPECT_FLOAT_EQ(c[k], a[k] + b[k]) << ", at index " << k;
    }
  }
}

TEST_P(ThreadingUtilsTestFixture, ParallelFor2D) {
  const int max_thread = treelite::threading_utils::MaxNumThread();
  auto sched = GetParam();

  /* Property-based testing with randomly generated parameters */
  constexpr int kRow = 100;
  constexpr int kCol = 100;
  constexpr int kElem = kRow * kCol;

  auto ind = [&](auto i, auto j) { return i * kRow + j; };

  RandomGenerator rng;
  std::vector<double> a(kElem);
  std::vector<double> b(kElem);
  std::generate_n(a.begin(), kElem, [&rng]() { return rng.DrawReal(-1.0, 1.0); });
  std::generate_n(b.begin(), kElem, [&rng]() { return rng.DrawReal(-10.0, 10.0); });

  constexpr int kNumTrial = 200;
  for (int trial_id = 0; trial_id < kNumTrial; ++trial_id) {
    std::vector<double> c(kElem);
    // Fill c with dummy values
    std::generate_n(c.begin(), kElem, [&rng]() { return rng.DrawReal(100.0, 200.0); });

    // Compute c := a + b on range [dim1_begin, dim1_end) x [dim2_begin, dim2_end)
    std::int64_t dim1_begin = rng.DrawInteger(0, kRow);
    std::int64_t dim1_end = rng.DrawInteger(dim1_begin, kRow);
    std::int64_t dim2_begin = rng.DrawInteger(0, kCol);
    std::int64_t dim2_end = rng.DrawInteger(dim2_begin, kCol);
    auto thread_config = threading_utils::ConfigureThreadConfig(
        static_cast<int>(rng.DrawInteger(1, max_thread + 1)));

    ParallelFor2D(
        dim1_begin, dim1_end, dim2_begin, dim2_end, thread_config, sched,
        [&](std::int64_t i, std::int64_t j, int) {
          c[ind(i, j)] = a[ind(i, j)] + b[ind(i, j)];
      });

    for (std::int64_t i = dim1_begin; i < dim1_end; ++i) {
      for (std::int64_t j = dim2_begin; j < dim2_end; ++j) {
        EXPECT_FLOAT_EQ(c[ind(i, j)], a[ind(i, j)] + b[ind(i, j)])
          << ", at index (" << i << ", " << j << ")";
      }
    }
  }
}

std::vector<ParallelSchedule> params = {
  ParallelSchedule::Auto(),
  ParallelSchedule::Static(),
  ParallelSchedule::Static(100),
  ParallelSchedule::Dynamic(),
  ParallelSchedule::Dynamic(100),
  ParallelSchedule::Guided(),
};

INSTANTIATE_TEST_SUITE_P(ThreadingUtils, ThreadingUtilsTestFixture, ::testing::ValuesIn(params));

}  // namespace threading_utils
}  // namespace treelite
