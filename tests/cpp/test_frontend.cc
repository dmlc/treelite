/*!
 * Copyright (c) 2020 by Contributors
 * \file test_frontend.cc
 * \author Hyunsu Cho
 * \brief C++ tests for frontends
 */
#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

#include "xgboost/xgboost_json.h"

using namespace fmt::literals;

namespace treelite {

class MockDelegator : public details::Delegator {
 public:
  MOCK_METHOD(void, pop_delegate, (), (override));
  MOCK_METHOD(
      void, push_delegate, (std::shared_ptr<details::BaseHandler> new_delegate), (override));
  rapidjson::Document const& get_handler_config() override {
    return mock_handler_config_;
  }

 private:
  rapidjson::Document mock_handler_config_;
};

class MockArrayStarter : public details::BaseHandler {
 public:
  MockArrayStarter(
      std::weak_ptr<details::Delegator> parent_delegator, details::BaseHandler& handler)
      : details::BaseHandler{parent_delegator}, wrapped_handler{handler} {};
  bool Null() {
    return wrapped_handler.Null();
  }
  bool Bool(bool b) {
    return wrapped_handler.Bool(b);
  }
  bool Int(int i) {
    return wrapped_handler.Int(i);
  }
  bool Uint(unsigned u) {
    return wrapped_handler.Uint(u);
  }
  bool Int64(int64_t i) {
    return wrapped_handler.Int64(i);
  }
  bool Uint64(uint64_t u) {
    return wrapped_handler.Uint64(u);
  }
  bool Double(double d) {
    return wrapped_handler.Double(d);
  }
  bool String(char const* str, std::size_t length, bool copy) {
    return wrapped_handler.String(str, length, copy);
  }
  bool StartObject() {
    return wrapped_handler.StartObject();
  }
  bool Key(char const* str, std::size_t length, bool copy) {
    return wrapped_handler.Key(str, length, copy);
  }
  bool EndObject(std::size_t memberCount) {
    return wrapped_handler.EndObject(memberCount);
  }
  bool StartArray() {
    return true;
  }
  bool EndArray(std::size_t elementCount) {
    return wrapped_handler.EndArray(elementCount);
  }

 private:
  details::BaseHandler& wrapped_handler;
};

class MockObjectStarter : public details::BaseHandler {
 public:
  MockObjectStarter(
      std::weak_ptr<details::Delegator> parent_delegator, details::BaseHandler& handler)
      : details::BaseHandler{parent_delegator}, wrapped_handler{handler} {};
  bool Null() {
    return wrapped_handler.Null();
  }
  bool Bool(bool b) {
    return wrapped_handler.Bool(b);
  }
  bool Int(int i) {
    return wrapped_handler.Int(i);
  }
  bool Uint(unsigned u) {
    return wrapped_handler.Uint(u);
  }
  bool Int64(int64_t i) {
    return wrapped_handler.Int64(i);
  }
  bool Uint64(uint64_t u) {
    return wrapped_handler.Uint64(u);
  }
  bool Double(double d) {
    return wrapped_handler.Double(d);
  }
  bool String(char const* str, std::size_t length, bool copy) {
    return wrapped_handler.String(str, length, copy);
  }
  bool StartObject() {
    return true;
  }
  bool Key(char const* str, std::size_t length, bool copy) {
    return wrapped_handler.Key(str, length, copy);
  }
  bool EndObject(std::size_t memberCount) {
    return wrapped_handler.EndObject(memberCount);
  }
  bool StartArray() {
    return wrapped_handler.StartArray();
  }
  bool EndArray(std::size_t elementCount) {
    return wrapped_handler.EndArray(elementCount);
  }

 private:
  details::BaseHandler& wrapped_handler;
};

/******************************************************************************
 * BaseHandler
 * ***************************************************************************/
class BaseHandlerFixture : public ::testing::TestWithParam<std::string> {};

TEST_P(BaseHandlerFixture, BaseHandler) {
  std::string json_str = GetParam();
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  details::BaseHandler handler{delegator};
  rapidjson::Reader reader;

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_FALSE(result);
}

INSTANTIATE_TEST_SUITE_P(BaseHandlerTests, BaseHandlerFixture,
    ::testing::Values("true", "-1", "1", "-4294967297", "4294967297", "0.1", "\"string\""));

TEST(BaseHandlerFixture, BaseHandlerObject) {
  std::string json_str = "{\"key\": 0}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  details::BaseHandler handler{delegator};
  rapidjson::Reader reader;

  EXPECT_CALL(*delegator, push_delegate).Times(0);
  EXPECT_CALL(*delegator, pop_delegate).Times(0);

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_FALSE(result);
}

TEST(BaseHandlerFixture, BaseHandlerArray) {
  std::string json_str = "[]";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  details::BaseHandler handler{delegator};
  rapidjson::Reader reader;

  EXPECT_CALL(*delegator, push_delegate).Times(0);
  EXPECT_CALL(*delegator, pop_delegate).Times(0);

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_FALSE(result);
}

/******************************************************************************
 * IgnoreHandler
 * ***************************************************************************/
class IgnoreHandlerFixture : public ::testing::TestWithParam<std::string> {};

TEST_P(IgnoreHandlerFixture, IgnoreHandlerPrims) {
  std::string json_str = GetParam();
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  details::IgnoreHandler handler{delegator};
  rapidjson::Reader reader;

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_TRUE(result);
}

INSTANTIATE_TEST_SUITE_P(IgnoreHandlerTests, IgnoreHandlerFixture,
    ::testing::Values("true", "-1", "1", "-4294967297", "4294967297", "0.1", "\"string\""));

TEST(IgnoreHandlerFixture, IgnoreHandlerObject) {
  std::string json_str = "{\"key\": 0}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  details::IgnoreHandler handler{delegator};
  rapidjson::Reader reader;

  EXPECT_CALL(*delegator, push_delegate).Times(1);
  EXPECT_CALL(*delegator, pop_delegate).Times(1);

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_TRUE(result);
}

TEST(IgnoreHandlerFixture, IgnoreHandlerArray) {
  std::string json_str = "[]";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  details::IgnoreHandler handler{delegator};
  rapidjson::Reader reader;

  EXPECT_CALL(*delegator, push_delegate).Times(1);
  EXPECT_CALL(*delegator, pop_delegate).Times(1);

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_TRUE(result);
}

/******************************************************************************
 * OutputHandler
 * ***************************************************************************/
class OutputHandlerFixture : public ::testing::TestWithParam<std::string> {};

TEST_P(OutputHandlerFixture, OutputHandler) {
  std::string json_str = GetParam();
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  int output;
  details::OutputHandler<int> handler{delegator, output};
  rapidjson::Reader reader;

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_FALSE(result);
}

INSTANTIATE_TEST_SUITE_P(OutputHandlerTests, OutputHandlerFixture,
    ::testing::Values("true", "-1", "1", "-4294967297", "4294967297", "0.1", "\"string\""));

TEST(OutputHandlerFixture, OutputHandlerObject) {
  std::string json_str = "{\"key\": 0}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  int output;
  details::OutputHandler<int> handler{delegator, output};
  rapidjson::Reader reader;

  EXPECT_CALL(*delegator, push_delegate).Times(0);
  EXPECT_CALL(*delegator, pop_delegate).Times(0);

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_FALSE(result);
}

TEST(OutputHandlerFixture, OutputHandlerArray) {
  std::string json_str = "[]";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();
  int output;
  details::OutputHandler<int> handler{delegator, output};
  rapidjson::Reader reader;

  EXPECT_CALL(*delegator, push_delegate).Times(0);
  EXPECT_CALL(*delegator, pop_delegate).Times(0);

  rapidjson::ParseResult result = reader.Parse(input_stream, handler);
  ASSERT_FALSE(result);
}

/******************************************************************************
 * ArrayHandler
 * ***************************************************************************/
TEST(ArrayHandlerSuite, ArrayHandler) {
  std::string json_str = "[0, 1, 2, 3]";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  std::vector<int> output;
  std::vector<int> expected{0, 1, 2, 3};

  details::ArrayHandler<int> wrapped_handler{delegator, output};
  MockArrayStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  rapidjson::ParseResult result = reader.Parse(input_stream, handler);

  ASSERT_TRUE(result);
  ASSERT_EQ(output, expected);
}

/******************************************************************************
 * TreeParamHandler
 * ***************************************************************************/
TEST(TreeParamHandlerSuite, TreeParamHandler) {
  std::string json_str
      = "{\"num_feature\": \"1\", \"num_nodes\": \"2\","
        " \"size_leaf_vector\": \"3\"}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  int output;
  int expected = 2;

  details::TreeParamHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  rapidjson::ParseResult result = reader.Parse(input_stream, handler);

  ASSERT_TRUE(result);
  ASSERT_EQ(output, expected);
}

/******************************************************************************
 * RegTreeHandler
 * ***************************************************************************/
TEST(RegTreeHandlerSuite, RegTreeHandler) {
  std::string json_str = "{\"loss_changes\": []}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  Tree<float, float> output;
  details::RegTreeHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  EXPECT_CALL(*delegator, push_delegate).Times(10);

  rapidjson::Reader reader;
  reader.Parse(input_stream, handler);

  json_str = "{\"sum_hessian\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"base_weights\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"leaf_child_counts\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"left_children\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"right_children\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"parents\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"split_indices\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"split_conditions\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);

  json_str = "{\"default_left\": []}";
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  reader.Parse(input_stream, handler);
}

TEST(RegTreeHandlerSuite, DefaultLeft) {
  class RegTreeHandlerWrapper : public details::BaseHandler {
   public:
    using BaseHandler::BaseHandler;
    bool StartObject() override {
      push_handler<details::RegTreeHandler, Tree<float, float>>(output);
      return true;
    }
    Tree<float, float> output;
  };
  rapidjson::Document handler_config;
  handler_config.Parse(R"({"allow_unknown_fields": false})");
  auto handler = details::DelegatedHandler::create_empty(handler_config);
  auto wrapped_handler = std::make_shared<RegTreeHandlerWrapper>(handler);
  handler->push_delegate(wrapped_handler);
  rapidjson::Reader reader;

  auto gather_default_left_array = [](Tree<float, float> const& tree) {
    std::vector<bool> default_left;
    for (int nid = 0; nid < tree.num_nodes; ++nid) {
      default_left.push_back(tree.DefaultLeft(nid));
    }
    return default_left;
  };

  // default_left array can be integers; Treelite should automatically cast
  // the elements to booleans.
  std::string json_str_fmt = R"JSON(
  {{"base_weights": [1.0, 0.0, 2.0, -0.5, 0.5, 1.5, 2.5],
    "default_left": [{default_left}],
    "id": 0,
    "left_children": [1, 3, 5, -1, -1, -1, -1],
    "loss_changes": [4.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    "parents": [2147483647, 0, 0, 1, 1, 2, 2],
    "right_children": [2, 4, 6, -1, -1, -1, -1],
    "split_conditions": [2.0, 1.0, 3.0, -0.5, 0.5, 1.5, 2.5],
    "split_indices": [0, 0, 0, 0, 0, 0, 0],
    "split_type": [0, 0, 0, 0, 0, 0, 0],
    "sum_hessian": [4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    "tree_param": {{
      "num_deleted": "0",
      "num_feature": "1",
      "num_nodes": "7",
      "size_leaf_vector": "0"}}
  }})JSON";
  std::string json_str = fmt::format(
      json_str_fmt, "default_left"_a = fmt::join(std::vector<int>{1, 1, 1, 0, 0, 0, 0}, ", "));
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  ASSERT_TRUE(reader.Parse(input_stream, *handler));
  std::vector<bool> expected_default_left{true, true, true, false, false, false, false};
  ASSERT_EQ(gather_default_left_array(wrapped_handler->output), expected_default_left);

  json_str = fmt::format(json_str_fmt,
      "default_left"_a
      = fmt::join(std::vector<bool>{true, true, true, false, false, false, false}, ", "));
  input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  ASSERT_TRUE(reader.Parse(input_stream, *handler));
  ASSERT_EQ(gather_default_left_array(wrapped_handler->output), expected_default_left);
}

/******************************************************************************
 * GBTreeModelHandler
 * ***************************************************************************/
TEST(GBTreeModelHandlerSuite, GBTreeModelHandler) {
  std::string json_str = "{\"trees\": []}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  details::ParsedXGBoostModel output{Model::Create<float, float>(), nullptr, {}, {}, ""};
  output.model = output.model_ptr.get();
  details::GBTreeModelHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  EXPECT_CALL(*delegator, push_delegate).Times(1);

  rapidjson::Reader reader;
  reader.Parse(input_stream, handler);
}

TEST(GBTreeModelHandlerSuite, TreeInfoField) {
  class GBTreeModelHandlerWrapper : public details::BaseHandler {
   public:
    using BaseHandler::BaseHandler;
    bool StartObject() override {
      push_handler<details::GBTreeModelHandler, details::ParsedXGBoostModel>(output);
      return true;
    }
    details::ParsedXGBoostModel output;
  };
  rapidjson::Document handler_config;
  handler_config.Parse(R"({"allow_unknown_fields": false})");
  auto handler = details::DelegatedHandler::create_empty(handler_config);
  auto wrapped_handler = std::make_shared<GBTreeModelHandlerWrapper>(handler);
  wrapped_handler->output.model_ptr = Model::Create<float, float>();
  wrapped_handler->output.model = wrapped_handler->output.model_ptr.get();
  handler->push_delegate(wrapped_handler);
  rapidjson::Reader reader;

  std::string json_str = R"({"trees": [], "tree_info": [0, 1, 2]})";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  ASSERT_TRUE(reader.Parse(input_stream, *handler));

  std::vector<int> expected_tree_info{0, 1, 2};
  ASSERT_EQ(wrapped_handler->output.tree_info, expected_tree_info);
}

/******************************************************************************
 * GradientBoosterHandler
 * ***************************************************************************/
TEST(GradientBoosterHandlerSuite, GradientBoosterHandler) {
  std::string json_str = "{\"name\": \"gbtree\"}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  details::ParsedXGBoostModel output{Model::Create<float, float>(), nullptr, {}, {}, ""};
  output.model = output.model_ptr.get();
  details::GradientBoosterHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  auto result = reader.Parse(input_stream, handler);

  ASSERT_TRUE(result);
}

/******************************************************************************
 * ObjectiveHandler
 * ***************************************************************************/
TEST(ObjectiveHandlerSuite, ObjectiveHandler) {
  std::string json_str = "{\"name\": \"multi:softmax\"}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  std::string output;
  details::ObjectiveHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  auto result = reader.Parse(input_stream, handler);

  ASSERT_TRUE(result);

  ASSERT_EQ(output, "multi:softmax");
}

/******************************************************************************
 * LearnerParamHandler
 * ***************************************************************************/
TEST(LearnerParamHandlerSuite, LearnerParamHandler) {
  std::string json_str = "{\"base_score\": \"0.5\", \"num_class\": \"5\", \"num_feature\": \"12\"}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  Model output;
  details::LearnerParamHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  auto result = reader.Parse(input_stream, handler);

  ASSERT_TRUE(result);

  ASSERT_FLOAT_EQ(output.param.global_bias, 0.5);
  ASSERT_EQ(output.task_param.num_class, 5);
  ASSERT_EQ(output.num_feature, 12);
}

/******************************************************************************
 * XGBoostModelHandler
 * ***************************************************************************/
TEST(XGBoostModelHandlerSuite, XGBoostModelHandler) {
  std::string json_str = "{\"version\": []}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  details::ParsedXGBoostModel output{Model::Create<float, float>(), nullptr, {}, {}, ""};
  output.model = output.model_ptr.get();
  details::XGBoostModelHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  reader.Parse(input_stream, handler);
  /* Merely testing that parsing does not error out. No non-trivial isolated
   * assertions are useful here. Full behavior tested in Python integration
   * test. */
}

/******************************************************************************
 * RootHandler
 * ***************************************************************************/
TEST(RootHandlerSuite, RootHandler) {
  std::string json_str = "{}";
  auto input_stream = rapidjson::MemoryStream(json_str.c_str(), json_str.size());
  std::shared_ptr<MockDelegator> delegator = std::make_shared<MockDelegator>();

  details::ParsedXGBoostModel output;
  details::RootHandler wrapped_handler{delegator, output};
  MockObjectStarter handler{delegator, wrapped_handler};

  rapidjson::Reader reader;
  reader.Parse(input_stream, handler);
  /* Merely testing that parsing does not error out. No non-trivial isolated
   * assertions are useful here. Full behavior tested in Python integration
   * test. */
}

}  // namespace treelite
