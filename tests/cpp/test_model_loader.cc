/*!
 * Copyright (c) 2023 by Contributors
 * \file test_model_loader.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model loader
 */

#include <gtest/gtest.h>
#include <model_loader/detail/string_utils.h>

#include <string>

TEST(ModelLoader, StringTrim) {
  std::string s{"foobar\r\n"};
  treelite::model_loader::detail::StringTrimFromEnd(s);
  EXPECT_EQ(s, "foobar");
}

TEST(ModelLoader, StringStartsWith) {
  std::string s{"foobar"};
  EXPECT_TRUE(treelite::model_loader::detail::StringStartsWith(s, "foo"));
}
