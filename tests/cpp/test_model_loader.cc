/*!
 * Copyright (c) 2023 by Contributors
 * \file test_model_loader.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model loader
 */

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "model_loader/detail/string_utils.h"
#include "model_loader/detail/xgboost.h"

TEST(ModelLoader, StringTrim) {
  std::string s{"foobar\r\n"};
  treelite::model_loader::detail::StringTrimFromEnd(s);
  EXPECT_EQ(s, "foobar");
}

TEST(ModelLoader, StringStartsWith) {
  std::string s{"foobar"};
  EXPECT_TRUE(treelite::model_loader::detail::StringStartsWith(s, "foo"));
}

TEST(ModelLoader, XGBoostBaseScore) {
  {
    std::string s{"[5.2008224E-1,4.665861E-1]"};
    std::vector<float> parsed = treelite::model_loader::detail::xgboost::ParseBaseScore(s);
    std::vector<float> expected{5.2008224E-1, 4.665861E-1};
    EXPECT_EQ(parsed, expected);
  }

  {
    std::string s{"4.9333417E-1"};
    std::vector<float> parsed = treelite::model_loader::detail::xgboost::ParseBaseScore(s);
    std::vector<float> expected{4.9333417E-1};
    EXPECT_EQ(parsed, expected);
  }
}
