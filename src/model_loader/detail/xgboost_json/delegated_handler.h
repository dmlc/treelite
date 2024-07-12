/*!
 * Copyright (c) 2020-2024 by Contributors
 * \file delegated_handler.h
 * \brief A delegated handler for loading XGBoost JSON/UBJSON models via SAX parsers.
 *        The delegated handler delegates the parsing of components to sub-handlers.
 * \author Hyunsu Cho, William Hicks
 */

#ifndef SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_DELEGATED_HANDLER_H_
#define SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_DELEGATED_HANDLER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stack>
#include <string>
#include <type_traits>
#include <vector>

#include <treelite/model_builder.h>

namespace treelite::model_loader::detail::xgboost {

class HandlerConfig {
 public:
  bool allow_unknown_field{false};
};

struct ParsedXGBoostModel {
  std::unique_ptr<treelite::model_builder::ModelBuilder> builder{};
  std::int32_t num_tree{0};
  std::vector<unsigned> version{};
  std::vector<int> tree_info{};
  std::string objective_name{};
  std::int32_t size_leaf_vector{0};
  std::vector<float> weight_drop{};
};

struct ParsedRegTreeParams {
  std::int32_t num_nodes{0};
  std::int32_t size_leaf_vector{0};
};

struct ParsedLearnerParams {
  float base_score{0.0};
  std::int32_t num_class{1};
  std::int32_t num_feature{0};
  std::int32_t num_target{1};
  bool boost_from_average{false};
};

class BaseHandler;
class RootHandler;

/*! \brief Class for handling delegation of JSON handling */
class Delegator {
 public:
  /*! \brief Pop stack of delegate handlers */
  virtual void pop_delegate() = 0;
  /*! \brief Push new delegate handler onto stack */
  virtual void push_delegate(std::shared_ptr<BaseHandler> new_delegate) = 0;
  /*! \brief Get configuration for handlers */
  virtual HandlerConfig const& get_handler_config() = 0;
};

/*! \brief Base handler class for parsing all JSON objects */
class BaseHandler {
 public:
  /*!
   * \brief Construct handler to be added to given delegator's stack
   * \param parent_delegator Pointer to Delegator for this handler
   */
  explicit BaseHandler(std::weak_ptr<Delegator> parent_delegator);

  virtual bool Null();
  virtual bool Bool(bool);
  virtual bool Int64(std::int64_t);
  virtual bool Uint64(std::uint64_t);
  virtual bool Double(double);
  virtual bool String(std::string const&);
  virtual bool StartObject();
  virtual bool Key(std::string const&);
  virtual bool EndObject();
  virtual bool StartArray();
  virtual bool EndArray();

 protected:
  /* \brief Build handler of indicated type and push it onto delegator's stack
   * \param args Any args required to build handler
   */
  template <typename HandlerType, typename... ArgsTypes>
  bool push_handler(ArgsTypes&... args) {
    if (auto parent = BaseHandler::delegator.lock()) {
      parent->push_delegate(std::make_shared<HandlerType>(delegator, args...));
      return true;
    } else {
      return false;
    }
  }

  /* \brief If current JSON key is the indicated string, build handler of
   *        indicated type and push it onto delegator's stack
   * \param key the expected key
   * \param args ... any args required to build handler
   */
  template <typename HandlerType, typename... ArgsTypes>
  bool push_key_handler(std::string const& key, ArgsTypes&... args) {
    if (check_cur_key(key)) {
      push_handler<HandlerType, ArgsTypes...>(args...);
      return true;
    } else {
      return false;
    }
  }

  /* \brief Pop handler off of delegator's stack, relinquishing parsing */
  bool pop_handler();
  /* \brief Store current JSON key
   * \param str Key to store
   * \return Whether the key is acceptable
   */
  bool set_cur_key(std::string const& key);
  /* \brief Retrieve current JSON key */
  std::string const& get_cur_key();
  /* \brief Check if current JSON key is indicated key
   * \param query_key Value to compare against current JSON key
   */
  bool check_cur_key(std::string const& query_key);
  /* \brief If current JSON key is the indicated string, assign value to output
   * \param key JSON key for this output
   * \param value Value to be assigned
   * \param output Reference to object to which the value should be assigned
   */
  template <typename ValueType>
  bool assign_value(std::string const& key, ValueType&& value, ValueType& output);
  template <typename ValueType>
  bool assign_value(std::string const& key, ValueType const& value, ValueType& output);

  /* \brief Check if a given key is recognized by the handler
   * \param key Key to look up
   */
  virtual bool is_recognized_key([[maybe_unused]] std::string const& key) {
    return false;
  }

  // Perform this check at the top of every value handling function,
  // to ignore extra fields from the JSON string
  virtual bool should_ignore_upcoming_value() {
    bool ret = state_next_field_ignore_;
    state_next_field_ignore_ = false;  // The state should be reset for every value token
    return ret;
  }

 private:
  /* \brief Delegator which delegated parsing responsibility to this handler */
  std::weak_ptr<Delegator> delegator;
  /* \brief JSON key for the object currently being parsed */
  std::string cur_key;
  /* \brief Whether to allow extra fields with unrecognized key; when false, extra fields
   *        will cause a fatal error. */
  bool allow_unknown_field_;
  /* \brief A boolean state, indicating whether the upcoming value should be ignored. This
   *        boolean will be set to true when an unknown key is encountered (and
   *        allow_unknown_field_ is set). */
  bool state_next_field_ignore_;
};

/*! \brief Handler which delegates JSON parsing to stack of delegates */
class DelegatedHandler : public Delegator {
 public:
  /*! \brief Create DelegatedHandler with empty stack
   *  \param handler_config Configuration to affect the behavior of handlers
   */
  static std::shared_ptr<DelegatedHandler> create_empty(HandlerConfig const& handler_config);

  /*! \brief Create DelegatedHandler with initial RootHandler on stack
   *  \param handler_config Configuration to affect the behavior of handlers
   */
  static std::shared_ptr<DelegatedHandler> create(HandlerConfig const& handler_config);

  /*! \brief Push new handler onto stack, delegating ongoing parsing to it
   *  \param new_delegate the delegate to push onto stack
   */
  void push_delegate(std::shared_ptr<BaseHandler> new_delegate) override;

  /*! \brief Pop handler off of stack, returning parsing responsibility to
   *         previous handler on stack
   */
  void pop_delegate() override;

  /*! \brief Query the handler configuration */
  HandlerConfig const& get_handler_config() override {
    return handler_config_;
  }

  /*! \brief Fetch the result of parsing */
  ParsedXGBoostModel get_result();

  bool Null();
  bool Bool(bool b);
  bool Int64(std::int64_t i);
  bool Uint64(std::uint64_t u);
  bool Double(double d);
  bool String(std::string const& str);
  bool StartObject();
  bool Key(std::string const& key);
  bool EndObject();
  bool StartArray();
  bool EndArray();

 private:
  explicit DelegatedHandler(HandlerConfig const& handler_config);

  std::stack<std::shared_ptr<BaseHandler>> delegates;
  ParsedXGBoostModel result;
  HandlerConfig const& handler_config_;
};

/*! \brief JSON handler that ignores all delegated input */
class IgnoreHandler : public BaseHandler {
 public:
  using BaseHandler::BaseHandler;
  bool Null() override;
  bool Bool(bool b) override;
  bool Int64(std::int64_t i) override;
  bool Uint64(std::uint64_t u) override;
  bool Double(double d) override;
  bool String(std::string const& str) override;
  bool StartObject() override;
  bool Key(std::string const& str) override;
  bool StartArray() override;
};

/*! \brief base handler for updating some output object*/
template <typename OutputType>
class OutputHandler : public BaseHandler {
 public:
  /*!
   * \brief Construct handler to be added to given delegator's stack
   * \param parent_delegator Pointer to Delegator for this handler
   * \param output Object to be modified during parsing
   */
  OutputHandler(std::weak_ptr<Delegator> parent_delegator, OutputType& output)
      : BaseHandler{parent_delegator}, output{output} {};
  OutputHandler(std::weak_ptr<Delegator> parent_delegator, OutputType&& output) = delete;

 protected:
  /* \brief Output value constructed or modified during parsing */
  OutputType& output;
};

/*! \brief Handler for array of objects of given type */
template <typename ElemType, typename HandlerType = BaseHandler>
class ArrayHandler : public OutputHandler<std::vector<ElemType>> {
 public:
  using OutputHandler<std::vector<ElemType>>::OutputHandler;

  /* Note: This method will only be instantiated (and therefore override the
   * base `bool Bool(bool)` method) if ElemType is bool. */
  bool Bool(ElemType b) {
    if (this->should_ignore_upcoming_value()) {
      return true;
    }
    this->output.push_back(b);
    return true;
  }

  template <typename ArgType, typename IntType = ElemType>
  std::enable_if_t<std::is_integral_v<IntType>, bool> store_int(ArgType i) {
    this->output.push_back(static_cast<ElemType>(i));
    return true;
  }

  template <typename ArgType, typename IntType = ElemType>
  std::enable_if_t<!std::is_integral_v<IntType>, bool> store_int(ArgType) {
    return false;
  }

  bool Int64(std::int64_t i) override {
    if (this->should_ignore_upcoming_value()) {
      return true;
    }
    return store_int<std::int64_t>(i);
  }

  bool Uint64(std::uint64_t u) override {
    if (this->should_ignore_upcoming_value()) {
      return true;
    }
    return store_int<std::uint64_t>(u);
  }

  bool Double(double d) {
    if (this->should_ignore_upcoming_value()) {
      return true;
    }
    if constexpr (!std::is_floating_point_v<ElemType>) {
      return false;
    } else {
      this->output.push_back(static_cast<ElemType>(d));
    }
    return true;
  }

  template <typename StringType = ElemType>
  typename std::enable_if_t<std::is_same_v<StringType, std::string>, bool> store_string(
      std::string const& str) {
    this->output.push_back(str);
    return true;
  }

  template <typename StringType = ElemType>
  typename std::enable_if_t<!std::is_same_v<StringType, std::string>, bool> store_string(
      std::string const&) {
    return false;
  }

  bool String(std::string const& str) override {
    if (this->should_ignore_upcoming_value()) {
      return true;
    }
    return store_string(str);
  }

  bool StartObject(std::true_type) {
    if (this->should_ignore_upcoming_value()) {
      return this->template push_handler<IgnoreHandler>();
    }
    this->output.emplace_back();
    return this->template push_handler<HandlerType, ElemType>(this->output.back());
  }

  bool StartObject(std::false_type) {
    return false;
  }

  bool StartObject() override {
    return StartObject(std::integral_constant<bool,
        std::is_base_of<OutputHandler<ElemType>, HandlerType>::value>{});
  }
};

/*! \brief Handler for TreeParam objects from XGBoost schema */
class TreeParamHandler : public OutputHandler<ParsedRegTreeParams> {
 public:
  using OutputHandler<ParsedRegTreeParams>::OutputHandler;

  bool String(std::string const& str) override;

 protected:
  bool is_recognized_key(std::string const& key) override;
};

/*! \brief Handler for RegTree objects from XGBoost schema */
class RegTreeHandler : public OutputHandler<ParsedRegTreeParams> {
 public:
  RegTreeHandler(std::weak_ptr<Delegator> parent_delegator, ParsedRegTreeParams& output,
      model_builder::ModelBuilder& model_builder);
  RegTreeHandler(std::weak_ptr<Delegator> parent_delegator, ParsedRegTreeParams&& output) = delete;

  bool StartArray() override;
  bool StartObject() override;
  bool Int64(std::int64_t) override;
  bool Uint64(std::uint64_t) override;
  bool EndObject() override;

 protected:
  bool is_recognized_key(std::string const& key) override;

 private:
  std::vector<float> loss_changes;
  std::vector<float> sum_hessian;
  std::vector<float> base_weights;
  std::vector<int> left_children;
  std::vector<int> right_children;
  std::vector<int> parents;
  std::vector<int> split_indices;
  std::vector<int> split_type;
  std::vector<int> categories_segments;
  std::vector<int> categories_sizes;
  std::vector<int> categories_nodes;
  std::vector<int> categories;
  std::vector<float> split_conditions;
  std::vector<bool> default_left;
  model_builder::ModelBuilder& model_builder;
};

/*! \brief Handler for array of objects of Tree type */
class RegTreeArrayHandler : public OutputHandler<std::vector<ParsedRegTreeParams>> {
 public:
  RegTreeArrayHandler(std::weak_ptr<Delegator> parent_delegator,
      std::vector<ParsedRegTreeParams>& output, model_builder::ModelBuilder& model_builder);
  RegTreeArrayHandler(
      std::weak_ptr<Delegator> parent_delegator, std::vector<ParsedRegTreeParams>&& output)
      = delete;

  bool StartObject() override;

 private:
  model_builder::ModelBuilder& model_builder;
};

/*! \brief Handler for GBTreeModel objects from XGBoost schema */
class GBTreeModelHandler : public OutputHandler<ParsedXGBoostModel> {
 public:
  using OutputHandler<ParsedXGBoostModel>::OutputHandler;
  bool StartArray() override;
  bool StartObject() override;
  bool EndObject() override;

 protected:
  bool is_recognized_key(std::string const& key) override;

 private:
  std::vector<ParsedRegTreeParams> reg_tree_params;
};

/*! \brief Handler for GradientBoosterHandler objects from XGBoost schema */
class GradientBoosterHandler : public OutputHandler<ParsedXGBoostModel> {
 public:
  using OutputHandler<ParsedXGBoostModel>::OutputHandler;
  bool String(std::string const& str) override;
  bool StartArray() override;
  bool StartObject() override;
  bool EndObject() override;

 protected:
  bool is_recognized_key(std::string const& key) override;

 private:
  std::string name;
  std::vector<float> weight_drop;
};

/*! \brief Handler for ObjectiveHandler objects from XGBoost schema */
class ObjectiveHandler : public OutputHandler<std::string> {
 public:
  using OutputHandler<std::string>::OutputHandler;
  bool StartObject() override;
  bool String(std::string const& str) override;

 protected:
  bool is_recognized_key(std::string const& key) override;
};

/*! \brief Handler for LearnerParam objects from XGBoost schema */
class LearnerParamHandler : public OutputHandler<ParsedLearnerParams> {
 public:
  using OutputHandler<ParsedLearnerParams>::OutputHandler;
  bool String(std::string const& str) override;

 protected:
  bool is_recognized_key(std::string const& key) override;
};

/*! \brief Handler for Learner objects from XGBoost schema */
class LearnerHandler : public OutputHandler<ParsedXGBoostModel> {
 public:
  using OutputHandler<ParsedXGBoostModel>::OutputHandler;
  bool StartObject() override;
  bool EndObject() override;
  bool StartArray() override;

 protected:
  bool is_recognized_key(std::string const& key) override;

 private:
  ParsedLearnerParams learner_params;
  std::string objective;
};

/*! \brief Handler for XGBoost checkpoint */
class XGBoostCheckpointHandler : public OutputHandler<ParsedXGBoostModel> {
 public:
  using OutputHandler<ParsedXGBoostModel>::OutputHandler;
  bool StartArray() override;
  bool StartObject() override;

 protected:
  bool is_recognized_key(std::string const& key) override;
};

/*! \brief Handler for XGBoostModel objects from XGBoost schema */
class XGBoostModelHandler : public OutputHandler<ParsedXGBoostModel> {
 public:
  using OutputHandler<ParsedXGBoostModel>::OutputHandler;
  bool StartArray() override;
  bool StartObject() override;

 protected:
  bool is_recognized_key(std::string const& key) override;
};

/*! \brief Handler for root object of XGBoost schema */
class RootHandler : public OutputHandler<ParsedXGBoostModel> {
 public:
  using OutputHandler<ParsedXGBoostModel>::OutputHandler;
  bool StartObject() override;
};

}  // namespace treelite::model_loader::detail::xgboost

#endif  // SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_DELEGATED_HANDLER_H_
