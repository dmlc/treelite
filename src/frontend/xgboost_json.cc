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
#include <fmt/format.h>
#include <rapidjson/error/en.h>
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
#include <string>
#include <utility>

#include "xgboost/xgboost.h"

namespace {

template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
                                             ErrorHandlerFunc error_handler);

inline FILE* OpenFile(const char* filename) {
#ifdef _WIN32
  FILE* fp = std::fopen(filename, "rb");
#else
  FILE* fp = std::fopen(filename, "r");
#endif
  CHECK(fp) << "Failed to open file '" << filename << "': " << std::strerror(errno);
  return fp;
}

class OwningFileReadStream {
 public:
  explicit OwningFileReadStream(const char* filename)
      : fp_(OpenFile(filename)), file_stream_(nullptr), read_buffer_("") {
    file_stream_ = std::make_unique<rapidjson::FileReadStream>(
        fp_, read_buffer_, sizeof(read_buffer_));
  }
  virtual ~OwningFileReadStream() {
    if (fp_) {
      LOG(INFO) << "Closing file";
      std::fclose(fp_);
    }
  }

  using Ch = rapidjson::FileReadStream::Ch;

  Ch Peek() const {
    return file_stream_->Peek();
  }
  Ch Take() {
    return file_stream_->Take();
  }
  size_t Tell() const {
    return file_stream_->Tell();
  }
  void Put(Ch c) {
    file_stream_->Put(c);
  }
  void Flush() {
    file_stream_->Flush();
  }
  Ch* PutBegin() {
    return file_stream_->PutBegin();
  }
  size_t PutEnd(Ch* c) {
    return file_stream_->PutEnd(c);
  }
  const Ch* Peek4() const {
    return file_stream_->Peek4();
  }

 private:
  FILE* fp_;
  std::unique_ptr<rapidjson::FileReadStream> file_stream_;
  char read_buffer_[65536];
};

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(xgboost_json);

std::unique_ptr<treelite::Model> LoadXGBoostJSONModel(const char* filename) {
  auto input_stream = std::make_unique<OwningFileReadStream>(filename);
  auto error_handler = [filename](size_t offset) -> std::string {
    FILE* fp = OpenFile(filename);
    size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::fseek(fp, cur, SEEK_SET);
    int c;
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      c = std::fgetc(fp);
      if (c == EOF) {
        break;
      }
      oss << static_cast<char>(c);
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    std::fclose(fp);
    return oss.str() + "\n" + oss2.str();
  };
  auto parsed_model = ParseStream(std::move(input_stream), error_handler);
  return parsed_model;
}

std::unique_ptr<treelite::Model> LoadXGBoostJSONModelString(const char* json_str, size_t length) {
  auto input_stream = std::make_unique<rapidjson::MemoryStream>(json_str, length);
  auto error_handler = [](size_t offset) -> std::string {
    return "";
  };
  return ParseStream(std::move(input_stream), error_handler);
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
          assign_value("num_nodes", atoi(str), output) ||
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
  output.Init();
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
      output.SetLeaf(new_id, split_conditions[old_id]);
    } else {
      output.AddChilds(new_id);
      output.SetNumericalSplit(
          new_id, split_indices[old_id], split_conditions[old_id],
          default_left[old_id], false, treelite::Operator::kLT);
      output.SetGain(new_id, loss_changes[old_id]);
      Q.push({left_children[old_id], output.LeftChild(new_id)});
      Q.push({right_children[old_id], output.RightChild(new_id)});
    }
    output.SetSumHess(new_id, sum_hessian[old_id]);
  }
  return pop_handler();
}

/******************************************************************************
 * GBTreeHandler
 * ***************************************************************************/
bool GBTreeModelHandler::StartArray() {
  return (push_key_handler<ArrayHandler<treelite::Tree<float, float>, RegTreeHandler>,
                           std::vector<treelite::Tree<float, float>>>(
                               "trees", output.trees) ||
          push_key_handler<IgnoreHandler>("tree_info"));
}

bool GBTreeModelHandler::StartObject() {
  return push_key_handler<IgnoreHandler>("gbtree_model_param");
}

/******************************************************************************
 * GradientBoosterHandler
 * ***************************************************************************/
bool GradientBoosterHandler::String(const char *str,
                                    std::size_t length,
                                    bool copy) {
  if (!check_cur_key("name")) {
    return false;
  }
  fmt::string_view name{str, length};
  if (name != "gbtree") {
    LOG(ERROR) << "Only GBTree-type boosters are currently supported.";
    return false;
  } else {
    return true;
  }
}
bool GradientBoosterHandler::StartObject() {
  if (push_key_handler<GBTreeModelHandler, treelite::ModelImpl<float, float>>("model", output)) {
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
  return assign_value("name", std::string{str, length}, output);
}

/******************************************************************************
 * LearnerParamHandler
 * ***************************************************************************/
bool LearnerParamHandler::String(const char *str,
                                 std::size_t length,
                                 bool copy) {
  return (assign_value("base_score", strtof(str, nullptr),
                       output.param.global_bias) ||
          assign_value("num_class", static_cast<int>(std::max(atoi(str), 1)),
                       output.num_output_group) ||
          assign_value("num_feature", atoi(str), output.num_feature));
}

/******************************************************************************
 * LearnerHandler
 * ***************************************************************************/
bool LearnerHandler::StartObject() {
  // "attributes" key is not documented in schema
  return (push_key_handler<LearnerParamHandler, treelite::ModelImpl<float, float>>(
              "learner_model_param", *output.model) ||
          push_key_handler<GradientBoosterHandler, treelite::ModelImpl<float, float>>(
              "gradient_booster", *output.model) ||
          push_key_handler<ObjectiveHandler, std::string>("objective", objective) ||
          push_key_handler<IgnoreHandler>("attributes"));
}

bool LearnerHandler::EndObject(std::size_t memberCount) {
  xgboost::SetPredTransform(objective, &output.model->param);
  output.objective_name = objective;
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
  return push_key_handler<LearnerHandler, XGBoostModelHandle>("learner", output);
}

bool XGBoostModelHandler::EndObject(std::size_t memberCount) {
  if (memberCount != 2) {
    return false;
  }
  output.model->random_forest_flag = false;
  // Before XGBoost 1.0.0, the global bias saved in model is a transformed value.  After
  // 1.0 it's the original value provided by user.
  const bool need_transform_to_margin = (version[0] >= 1);
  if (need_transform_to_margin) {
    treelite::details::xgboost::TransformGlobalBiasToMargin(
        output.objective_name, &output.model->param);
  }
  return pop_handler();
}

/******************************************************************************
 * RootHandler
 * ***************************************************************************/
bool RootHandler::StartObject() {
  handle = {dynamic_cast<treelite::ModelImpl<float, float>*>(output.get()), ""};
  return push_handler<XGBoostModelHandler, XGBoostModelHandle>(handle);
}

/******************************************************************************
 * DelegatedHandler
 * ***************************************************************************/
std::unique_ptr<treelite::Model> DelegatedHandler::get_result() { return std::move(result); }
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
template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
                                             ErrorHandlerFunc error_handler) {
  std::shared_ptr<treelite::details::DelegatedHandler> handler =
    treelite::details::DelegatedHandler::create();
  rapidjson::Reader reader;

  rapidjson::ParseResult result = reader.Parse(*input_stream, *handler);
  if (!result) {
    const auto error_code = result.Code();
    const size_t offset = result.Offset();
    input_stream.reset();

    std::string diagnostic = error_handler(offset);

    LOG(FATAL) << "Provided JSON could not be parsed as XGBoost model. Parsing error at offset "
               << offset << ": " << rapidjson::GetParseError_En(error_code) << "\n"
               << diagnostic;
  }
  return handler->get_result();
}
}  // anonymous namespace
