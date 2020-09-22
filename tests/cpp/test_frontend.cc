/*!
 * Copyright (c) 2020 by Contributors
 * \file test_frontend.cc
 * \author Hyunsu Cho
 * \brief C++ tests for frontends
 */
#include <gtest/gtest.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>

namespace treelite {

TEST(XGBoostJSONFrontend, Basic) {
  std::string json_str = "{}";

  Model model;
  frontend::LoadXGBoostJSONModelString(json_str, &model);
}

}  // namespace treelite
