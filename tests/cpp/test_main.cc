/*!
 * Copyright (c) 2020 by Contributors
 * \file test_main.cc
 * \author Hyunsu Cho
 * \brief Launcher for C++ unit tests, using Google Test framework
 */
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
