/*!
 * Copyright (c) 2021 by Contributors
 * \file test_compiler_param.cc
 * \author Hyunsu Cho
 * \brief C++ tests for ingesting compiler parameters
 */
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <treelite/compiler_param.h>
#include <treelite/logging.h>
#include <fmt/format.h>

using namespace testing;

namespace treelite {
namespace compiler {

TEST(CompilerParam, Basic) {
  std::string json_str = R"JSON(
    {
      "quantize": 1,
      "parallel_comp": 100,
      "native_lib_name": "predictor",
      "annotate_in": "annotation.json",
      "verbose": 3,
      "code_folding_req": 1.0,
      "dump_array_as_elf": 0
    })JSON";
  CompilerParam param = CompilerParam::ParseFromJSON(json_str.c_str());
  EXPECT_EQ(param.quantize, 1);
  EXPECT_EQ(param.parallel_comp, 100);
  EXPECT_EQ(param.native_lib_name, "predictor");
  EXPECT_EQ(param.annotate_in, "annotation.json");
  EXPECT_EQ(param.verbose, 3);
  EXPECT_EQ(param.code_folding_req, 1.0);
  EXPECT_EQ(param.dump_array_as_elf, 0);
}

TEST(CompilerParam, NonExistentKey) {
  std::string json_str = R"JSON(
    {
      "quantize": 1,
      "parallel_comp": 100,
      "nonexistent": 0.3
    })JSON";
  EXPECT_THAT(
      [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
      ThrowsMessage<treelite::Error>(HasSubstr("Unrecognized key 'nonexistent'")));
  json_str = R"JSON(
    {
      "quantize": 1,
      "parallel_comp": 100,
      "extra_object": {
        "extra": 30
      }
    })JSON";
  EXPECT_THAT(
      [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
      ThrowsMessage<treelite::Error>(HasSubstr("Unrecognized key 'extra_object'")));
}

TEST(CompilerParam, IncorrectType) {
  using namespace testing;
  std::string json_str = R"JSON(
    {
      "quantize": "bad_type"
    })JSON";
  EXPECT_THAT(
      [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
      ThrowsMessage<treelite::Error>(HasSubstr("Expected an integer for 'quantize'")));
  json_str = R"JSON(
    {
      "code_folding_req": {
        "bad_type": 30
      }
    })JSON";
  EXPECT_THAT(
      [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
      ThrowsMessage<treelite::Error>(
        HasSubstr("Expected a floating-point decimal for 'code_folding_req'")));
  json_str = R"JSON(
    {
      "native_lib_name": -10.0
    })JSON";
  EXPECT_THAT(
      [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
      ThrowsMessage<treelite::Error>(HasSubstr("Expected a string for 'native_lib_name'")));
  json_str = R"JSON(
    {
      "code_folding_req": 13bad
    })JSON";
  EXPECT_THAT(
      [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
      ThrowsMessage<treelite::Error>(HasSubstr("Got an invalid JSON string")));
}

TEST(CompilerParam, InvalidRange) {
  std::string json_str;
  for (const auto& key : std::vector<std::string>{"quantize", "parallel_comp",
                                                  "code_folding_req", "dump_array_as_elf"}) {
    std::string literal = (key == "code_folding_req" ? "-1.0" : "-1");
    json_str = fmt::format(R"JSON({{ "{0}": {1} }})JSON", key, literal);
    std::string expected_error = fmt::format("'{}' must be 0 or greater", key);
    EXPECT_THAT(
        [&]() { CompilerParam::ParseFromJSON(json_str.c_str()); },
        ThrowsMessage<treelite::Error>(HasSubstr(expected_error)));
  }
}

}  // namespace compiler
}  // namespace treelite
