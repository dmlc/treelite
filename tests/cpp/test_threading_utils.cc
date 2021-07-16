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
#include <thread>
#include <random>
#include <cstddef>
#include <cstdint>
#include "threading_utils/parallel_for.h"

using namespace testing;

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

TEST(ThreadingUtils, ComputeWorkRange) {
  /* Test error handling */
  EXPECT_THAT([&]() { ComputeWorkRange(0, 100, 0); },
              ThrowsMessage<treelite::Error>(HasSubstr("nthread must be positive")));
  EXPECT_THAT([&]() { ComputeWorkRange(-100, 100, 3); },
              ThrowsMessage<treelite::Error>(HasSubstr("begin must be 0 or greater")));
  EXPECT_THAT([&]() { ComputeWorkRange(-200, -100, 3); },
              ThrowsMessage<treelite::Error>(HasSubstr("end must be 0 or greater")));
  EXPECT_THAT([&]() { ComputeWorkRange(200, 100, 3); },
              ThrowsMessage<treelite::Error>(HasSubstr("end cannot be less than begin")));

  /* Property-based testing with randomly generated parameters */
  RandomGenerator rng;

  constexpr int kNumTrial = 200;
  for (int i = 0; i < kNumTrial; ++i) {
    int64_t begin = rng.DrawInteger(0, 10000);
    std::size_t nthread = static_cast<std::size_t>(rng.DrawInteger(1, 100));
    int64_t end = rng.DrawInteger(begin, 10000);
    auto range = ComputeWorkRange(begin, end, nthread);
    EXPECT_EQ(range.size(), nthread + 1);
    EXPECT_EQ(range[0], begin);
    EXPECT_EQ(range[nthread], end);
    for (std::size_t i = 0; i < nthread; ++i) {
      EXPECT_GE(range[i + 1], range[i]);
    }
  }
  // Test the case with begin == end
  for (int i = 0; i < 10; ++i) {
    int64_t begin = rng.DrawInteger(0, 10000);
    int64_t end = begin;
    std::size_t nthread = static_cast<std::size_t>(rng.DrawInteger(1, 100));
    auto range = ComputeWorkRange(begin, end, nthread);
    EXPECT_EQ(range.size(), nthread + 1);
    EXPECT_EQ(range[0], begin);
    EXPECT_EQ(range[nthread], begin);
    for (std::size_t i = 0; i < nthread; ++i) {
      EXPECT_EQ(range[i + 1], range[i]);
    }
  }
}

TEST(ThreadingUtils, ParallelFor) {
  /* Test error handling */
  const int max_thread = std::thread::hardware_concurrency();

  auto dummy_func = [](int, std::size_t) {};
  EXPECT_THAT([&]() { ParallelFor(0, 100, 0, dummy_func); },
              ThrowsMessage<treelite::Error>(HasSubstr("nthread must be positive")));
  EXPECT_THAT([&]() { ParallelFor(200, 100, 3, dummy_func); },
              ThrowsMessage<treelite::Error>(HasSubstr("end cannot be less than begin")));
  EXPECT_THAT([&]() { ParallelFor(-100, 100, 3, dummy_func); },
              ThrowsMessage<treelite::Error>(HasSubstr("begin must be 0 or greater")));
  EXPECT_THAT([&]() { ParallelFor(-200, -100, 3, dummy_func); },
              ThrowsMessage<treelite::Error>(HasSubstr("end must be 0 or greater")));
  EXPECT_THAT([&]() { ParallelFor(200, 100, 3, dummy_func); },
              ThrowsMessage<treelite::Error>(HasSubstr("end cannot be less than begin")));
  EXPECT_THAT([&]() { ParallelFor(10, 20, 3 * max_thread, dummy_func); },
              ThrowsMessage<treelite::Error>(HasSubstr(
                  std::string("nthread cannot exceed ") + std::to_string(max_thread))));

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

    ParallelFor(begin, end, nthread, [&a, &b, &c](int64_t i, std::size_t) {
      c[i] = a[i] + b[i];
    });

    for (int64_t i = begin; i < end; ++i) {
      EXPECT_FLOAT_EQ(c[i], a[i] + b[i]) << ", at index " << i;
    }
  }
}

}  // namespace threading_utils
}  // namespace treelite
