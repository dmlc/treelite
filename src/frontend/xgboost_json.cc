/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_json.cc
 * \brief Frontend for xgboost model
 * \author Hyunsu Cho
 * \author William Hicks
 */

#include "xgboost/xgboost_json.h"

#include <dmlc/registry.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <fmt/format.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <stack>
#include <string>
#include <utility>

#include "xgboost/xgboost.h"

namespace {

template<typename StreamType> treelite::Model ParseStream(
    std::unique_ptr<StreamType> input_stream
);

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(xgboost_json);

void LoadXGBoostJSONModel(const char* filename, Model* out) {
  char readBuffer[65536];

#ifdef _WIN32
  FILE * fp = fopen(filename, "rb");
#else
  FILE * fp = fopen(filename, "r");
#endif

  auto input_stream = std::make_unique<rapidjson::FileReadStream> (
      fp,
      readBuffer,
      sizeof(readBuffer));
  *out = std::move(ParseStream(std::move(input_stream)));
  fclose(fp);
}

void LoadXGBoostJSONModelString(const char* json_str,
                                size_t length, Model *out) {
  auto input_stream = std::make_unique<rapidjson::MemoryStream>(
      json_str, length);
  *out = std::move(ParseStream(std::move(input_stream)));
}

}  // namespace frontend

namespace details {

/******************************************************************************
 * BaseHandler
 * ***************************************************************************/

bool BaseHandler::pop_handler() {
  if (auto parent = delegator.lock()) {
    parent->pop_delegate();
    return true;
  } else {
    return false;
  }
}

void BaseHandler::set_cur_key(const char *str, std::size_t length) {
  cur_key = std::string{str, length};
}

const std::string &BaseHandler::get_cur_key() { return cur_key; }

bool BaseHandler::check_cur_key(const std::string &query_key) {
  return cur_key == query_key;
}

template <typename ValueType>
bool BaseHandler::assign_value(const std::string &key,
                               ValueType &&value,
                               ValueType &output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

template <typename ValueType>
bool BaseHandler::assign_value(const std::string &key,
                  const ValueType &value,
                  ValueType &output) {
  if (check_cur_key(key)) {
    output = value;
    return true;
  } else {
    return false;
  }
}

/******************************************************************************
 * IgnoreHandler
 * ***************************************************************************/
bool IgnoreHandler::Null() { return true; }
bool IgnoreHandler::Bool(bool b) { return true; }
bool IgnoreHandler::Int(int i) { return true; }
bool IgnoreHandler::Uint(unsigned u) { return true; }
bool IgnoreHandler::Int64(int64_t i) { return true; }
bool IgnoreHandler::Uint64(uint64_t u) { return true; }
bool IgnoreHandler::Double(double d) { return true; }
bool IgnoreHandler::String(const char *str, std::size_t length, bool copy) {
  return true; }
bool IgnoreHandler::StartObject() { return push_handler<IgnoreHandler>(); }
bool IgnoreHandler::Key(const char *str, std::size_t length, bool copy) {
  return true; }
bool IgnoreHandler::StartArray() { return push_handler<IgnoreHandler>(); }

/******************************************************************************
 * TreeParamHandler
 * ***************************************************************************/
bool TreeParamHandler::String(const char *str, std::size_t length, bool copy) {
  // Key "num_deleted" deprecated but still present in some xgboost output
  return (check_cur_key("num_feature") ||
          assign_value("num_nodes", atoi(str), m_output) ||
          check_cur_key("size_leaf_vector") || check_cur_key("num_deleted"));
}

/******************************************************************************
 * RegTreeHandler
 * ***************************************************************************/
bool RegTreeHandler::StartArray() {
  /* Keys "categories" and "split_type" not currently documented in schema but
   * will be used for upcoming categorical split feature */
  return (
      push_key_handler<ArrayHandler<double>>("loss_changes", loss_changes) ||
      push_key_handler<ArrayHandler<double>>("sum_hessian", sum_hessian) ||
      push_key_handler<ArrayHandler<double>>("base_weights", base_weights) ||
      push_key_handler<IgnoreHandler>("categories") ||
      push_key_handler<ArrayHandler<int>>("leaf_child_counts",
                                          leaf_child_counts) ||
      push_key_handler<ArrayHandler<int>>("left_children", left_children) ||
      push_key_handler<ArrayHandler<int>>("right_children", right_children) ||
      push_key_handler<ArrayHandler<int>>("parents", parents) ||
      push_key_handler<ArrayHandler<int>>("split_indices", split_indices) ||
      push_key_handler<IgnoreHandler>("split_type") ||
      push_key_handler<ArrayHandler<double>>("split_conditions",
                                             split_conditions) ||
      push_key_handler<ArrayHandler<bool>>("default_left", default_left));
}

bool RegTreeHandler::StartObject() {
  return push_key_handler<TreeParamHandler, int>("tree_param", num_nodes);
}

bool RegTreeHandler::Uint(unsigned u) { return check_cur_key("id"); }

bool RegTreeHandler::EndObject(std::size_t memberCount) {
  m_output.Init();
  if (num_nodes != loss_changes.size() || num_nodes != sum_hessian.size() ||
      num_nodes != base_weights.size() ||
      num_nodes != leaf_child_counts.size() ||
      num_nodes != left_children.size() ||
      num_nodes != right_children.size() || num_nodes != parents.size() ||
      num_nodes != split_indices.size() ||
      num_nodes != split_conditions.size() ||
      num_nodes != default_left.size()) {
    return false;
  }

  std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
  Q.push({0, 0});
  while (!Q.empty()) {
    int old_id, new_id;
    std::tie(old_id, new_id) = Q.front();
    Q.pop();

    if (left_children[old_id] == -1) {
      m_output.SetLeaf(new_id, base_weights[old_id]);
    } else {
      m_output.AddChilds(new_id);
      m_output.SetNumericalSplit(
          new_id, split_indices[old_id], split_conditions[old_id],
          default_left[old_id], treelite::Operator::kLT);
      m_output.SetGain(new_id, loss_changes[old_id]);
      Q.push({left_children[old_id], m_output.LeftChild(new_id)});
      Q.push({right_children[old_id], m_output.RightChild(new_id)});
    }
    m_output.SetSumHess(new_id, sum_hessian[old_id]);
  }
  return pop_handler();
}

/******************************************************************************
 * GBTreeHandler
 * ***************************************************************************/
bool GBTreeModelHandler::StartArray() {
  return (push_key_handler<ArrayHandler<treelite::Tree, RegTreeHandler>,
                           std::vector<treelite::Tree>>("trees",
                                                        m_output.trees) ||
          push_key_handler<IgnoreHandler>("tree_info"));
}

bool GBTreeModelHandler::StartObject() {
  return push_key_handler<IgnoreHandler>("gbtree_model_param");
}

/******************************************************************************
 * GradientBoosterHandler
 * ***************************************************************************/
bool GradientBoosterHandler::Uint(unsigned u) {
  return check_cur_key("num_boosting_round");
}
bool GradientBoosterHandler::String(const char *str,
                                    std::size_t length,
                                    bool copy) {
  fmt::string_view name{str, length};
  if (name != "gbtree") {
    LOG(ERROR) << "Only GBTree-type boosters are currently supported.";
    return false;
  } else {
    return true;
  }
}
bool GradientBoosterHandler::StartObject() {
  if (push_key_handler<GBTreeModelHandler, treelite::Model>("model",
                                                            m_output)) {
    return true;
  } else {
    LOG(ERROR) << "Key \"" << get_cur_key()
               << "\" not recognized. Is this a GBTree-type booster?";
    return false;
  }
}

/******************************************************************************
 * ObjectiveHandler
 * ***************************************************************************/
bool ObjectiveHandler::StartObject() {
  return (push_key_handler<IgnoreHandler>("reg_loss_param") ||
          push_key_handler<IgnoreHandler>("poisson_regression_param") ||
          push_key_handler<IgnoreHandler>("tweedie_regression_param") ||
          push_key_handler<IgnoreHandler>("softmax_multiclass_param") ||
          push_key_handler<IgnoreHandler>("lambda_rank_param") ||
          push_key_handler<IgnoreHandler>("aft_loss_param"));
}

bool ObjectiveHandler::String(const char *str, std::size_t length, bool copy) {
  return assign_value("name", std::string{str, length}, m_output);
}

/******************************************************************************
 * LearnerParamHandler
 * ***************************************************************************/
bool LearnerParamHandler::String(const char *str,
                                 std::size_t length,
                                 bool copy) {
  return (assign_value("base_score", strtof(str, nullptr),
                       m_output.param.global_bias) ||
          assign_value("num_class", static_cast<int>(std::max(atoi(str), 1)),
                       m_output.num_output_group) ||
          assign_value("num_feature", atoi(str), m_output.num_feature));
}

/******************************************************************************
 * LearnerHandler
 * ***************************************************************************/
bool LearnerHandler::StartArray() {
  return push_key_handler<IgnoreHandler>("eval_metrics"); }

bool LearnerHandler::StartObject() {
  // "attributes" key is not documented in schema
  return (push_key_handler<LearnerParamHandler, treelite::Model>(
              "learner_model_param", m_output) ||
          push_key_handler<GradientBoosterHandler, treelite::Model>(
              "gradient_booster", m_output) ||
          push_key_handler<ObjectiveHandler, std::string>("objective",
                                                          objective) ||
          push_key_handler<IgnoreHandler>("attributes"));
}

bool LearnerHandler::EndObject(std::size_t memberCount) {
  if (objective == "multi:softmax") {
    std::strncpy(m_output.param.pred_transform, "max_index",
                 sizeof(m_output.param.pred_transform));
  } else if (objective == "multi:softprob") {
    std::strncpy(m_output.param.pred_transform, "softmax",
                 sizeof(m_output.param.pred_transform));
  } else if (objective == "reg:logistic" || objective == "binary:logistic") {
    std::strncpy(m_output.param.pred_transform, "sigmoid",
                 sizeof(m_output.param.pred_transform));
  } else if (objective == "count:poisson" || objective == "reg:gamma" ||
             objective == "reg:tweedie") {
    std::strncpy(m_output.param.pred_transform, "exponential",
                 sizeof(m_output.param.pred_transform));
  } else {
    std::strncpy(m_output.param.pred_transform, "identity",
                 sizeof(m_output.param.pred_transform));
  }
  return pop_handler();
}

/******************************************************************************
 * XGBoostModelHandler
 * ***************************************************************************/
bool XGBoostModelHandler::StartArray() {
  return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>(
      "version", version);
}

bool XGBoostModelHandler::StartObject() {
  return push_key_handler<LearnerHandler, treelite::Model>("learner",
                                                           m_output);
}

bool XGBoostModelHandler::EndObject(std::size_t memberCount) {
  if (memberCount != 2) {
    return false;
  }
  m_output.random_forest_flag = false;
  // Before XGBoost 1.0.0, the global bias saved in model is a transformed value.  After
  // 1.0 it's the original value provided by user.
  if (version[0] >= 1) {
    if (std::strcmp(m_output.param.pred_transform, "sigmoid") == 0) {
      m_output.param.global_bias =
          ProbToMargin::Sigmoid(m_output.param.global_bias);
    } else if (std::strcmp(m_output.param.pred_transform, "exponential") ==
               0) {
      m_output.param.global_bias =
          ProbToMargin::Exponential(m_output.param.global_bias);
    }
  }
  return pop_handler();
}

/******************************************************************************
 * RootHandler
 * ***************************************************************************/
bool RootHandler::StartObject() {
  return push_handler<XGBoostModelHandler, treelite::Model>(m_output);
}

/******************************************************************************
 * DelegatedHandler
 * ***************************************************************************/
treelite::Model DelegatedHandler::get_result() { return std::move(result); }
bool DelegatedHandler::Null() { return delegates.top()->Null(); }
bool DelegatedHandler::Bool(bool b) { return delegates.top()->Bool(b); }
bool DelegatedHandler::Int(int i) { return delegates.top()->Int(i); }
bool DelegatedHandler::Uint(unsigned u) { return delegates.top()->Uint(u); }
bool DelegatedHandler::Int64(int64_t i) { return delegates.top()->Int64(i); }
bool DelegatedHandler::Uint64(uint64_t u) { return delegates.top()->Uint64(u); }
bool DelegatedHandler::Double(double d) { return delegates.top()->Double(d); }
bool DelegatedHandler::String(const char *str, std::size_t length, bool copy) {
  return delegates.top()->String(str, length, copy);
}
bool DelegatedHandler::StartObject() { return delegates.top()->StartObject(); }
bool DelegatedHandler::Key(const char *str, std::size_t length, bool copy) {
  return delegates.top()->Key(str, length, copy);
}
bool DelegatedHandler::EndObject(std::size_t memberCount) {
  return delegates.top()->EndObject(memberCount);
}
bool DelegatedHandler::StartArray() { return delegates.top()->StartArray(); }
bool DelegatedHandler::EndArray(std::size_t elementCount) {
  return delegates.top()->EndArray(elementCount);
}


}  // namespace details
}  // namespace treelite

namespace {
template<typename StreamType>
treelite::Model ParseStream(std::unique_ptr<StreamType> input_stream) {
  std::shared_ptr<treelite::details::DelegatedHandler> handler = \
    treelite::details::DelegatedHandler::create();
  rapidjson::Reader reader;

  rapidjson::ParseResult result = reader.Parse(*input_stream, *handler);
  if (!result) {
    LOG(ERROR) << "Parsing error " << result.Code() << " at offset "
               << result.Offset();
    throw std::runtime_error(
        "Provided JSON could not be parsed as XGBoost model");
  }
  return handler->get_result();
}
}  // anonymous namespace
