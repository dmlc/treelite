/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_json.h
 * \brief Methods for loading models from XGBoost-style JSON
 * \author William Hicks
 */

#include <rapidjson/document.h>
#include <treelite/tree.h>

#include <memory>
#include <stack>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

#ifndef TREELITE_FRONTEND_XGBOOST_XGBOOST_JSON_H_
#define TREELITE_FRONTEND_XGBOOST_XGBOOST_JSON_H_
namespace treelite {
namespace details {

class BaseHandler;

/*! \brief class for handling delegation of JSON handling*/
class Delegator {
 public:
  /*! \brief pop stack of delegate handlers*/
  virtual void pop_delegate() = 0;
  /*! \brief push new delegate handler onto stack*/
  virtual void push_delegate(std::shared_ptr<BaseHandler> new_delegate) = 0;
};

/*! \brief base class for parsing all JSON objects*/
class BaseHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, BaseHandler> {
 public:
  /*!
   * \brief construct handler to be added to given delegator's stack
   * \param parent_delegator pointer to Delegator for this handler
   */
  explicit BaseHandler(std::weak_ptr<Delegator> parent_delegator) :
    delegator{parent_delegator} {};

  virtual bool Null() { return false; }
  virtual bool Bool(bool) { return false; }
  virtual bool Int(int) { return false; }
  virtual bool Uint(unsigned) { return false; }
  virtual bool Int64(std::int64_t) { return false; }
  virtual bool Uint64(std::uint64_t) { return false; }
  virtual bool Double(double) { return false; }
  virtual bool String(const char *, std::size_t, bool) {
    return false;
  }
  virtual bool StartObject() { return false; }
  virtual bool Key(const char *str, std::size_t length, bool) {
    set_cur_key(str, length);
    return true;
  }
  virtual bool EndObject(std::size_t) { return pop_handler(); }
  virtual bool StartArray() { return false; }
  virtual bool EndArray(std::size_t) { return pop_handler(); }

 protected:
  /* \brief build handler of indicated type and push it onto delegator's stack
   * \param args ... any args required to build handler
   */
  template <typename HandlerType, typename... ArgsTypes>
  bool push_handler(ArgsTypes &... args) {
    if (auto parent = BaseHandler::delegator.lock()) {
      parent->push_delegate(std::make_shared<HandlerType>(delegator, args...));
      return true;
    } else {
      return false;
    }
  }

  /* \brief if current JSON key is the indicated string, build handler of
   *        indicated type and push it onto delegator's stack
   * \param key the expected key
   * \param args ... any args required to build handler
   */
  template <typename HandlerType, typename... ArgsTypes>
  bool push_key_handler(std::string key, ArgsTypes &... args) {
    if (check_cur_key(key)) {
      push_handler<HandlerType, ArgsTypes...>(args...);
      return true;
    } else {
      return false;
    }
  }
  /* \brief pop handler off of delegator's stack, relinquishing parsing */
  bool pop_handler();
  /* \brief store current JSON key
   * \param str the key to store
   * \param length the length of the str char array
   */
  void set_cur_key(const char *str, std::size_t length);
  /* \brief retrieve current JSON key */
  const std::string &get_cur_key();
  /* \brief check if current JSON key is indicated key
   * \param query_key the value to compare against current JSON key
   */
  bool check_cur_key(const std::string &query_key);
  /* \brief if current JSON key is the indicated string, assign value to output
   * \param key the JSON key for this output
   * \param value the value to be assigned
   * \param output reference to object to which the value should be assigned
   */
  template <typename ValueType>
  bool assign_value(const std::string &key,
                    ValueType &&value,
                    ValueType &output);
  template <typename ValueType>
  bool assign_value(const std::string &key,
                    const ValueType &value,
                    ValueType &output);

 private:
  /* \brief the delegator which delegated parsing responsibility to this handler */
  std::weak_ptr<Delegator> delegator;
  /* \brief the JSON key for the object currently being parsed */
  std::string cur_key;
};

/*! \brief JSON handler that ignores all delegated input*/
class IgnoreHandler : public BaseHandler {
 public:
  using BaseHandler::BaseHandler;
  bool Null() override;
  bool Bool(bool b) override;
  bool Int(int i) override;
  bool Uint(unsigned u) override;
  bool Int64(std::int64_t i) override;
  bool Uint64(std::uint64_t u) override;
  bool Double(double d) override;
  bool String(const char *str, std::size_t length, bool copy) override;
  bool StartObject() override;
  bool Key(const char *str, std::size_t length, bool copy) override;
  bool StartArray() override;
};

/*! \brief base handler for updating some output object*/
template <typename OutputType> class OutputHandler : public BaseHandler {
 public:
  /*!
   * \brief construct handler to be added to given delegator's stack
   * \param parent_delegator pointer to Delegator for this handler
   * \param output the object to be modified during parsing
   */
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &output_param)
      : BaseHandler{parent_delegator}, output{output_param} {};
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &&output) = delete;

 protected:
  /* \brief the output value constructed or modified during parsing */
  OutputType &output;
};

/*! \brief handler for array of objects of given type*/
template <typename ElemType, typename HandlerType = BaseHandler>
class ArrayHandler : public OutputHandler<std::vector<ElemType>> {
 public:
  using OutputHandler<std::vector<ElemType>>::OutputHandler;

  /* Note: This method will only be instantiated (and therefore override the
   * base `bool Bool(bool)` method) if ElemType is bool. */
  bool Bool(ElemType b) {
    this->output.push_back(b);
    return true;
  }
  template <typename ArgType, typename IntType = ElemType>
  typename std::enable_if<std::is_integral<IntType>::value, bool>::type
  store_int(ArgType i) {
    this->output.push_back(static_cast<ElemType>(i));
    return true;
  }

  template <typename ArgType, typename IntType = ElemType>
  typename std::enable_if<!std::is_integral<IntType>::value, bool>::type
  store_int(ArgType) {
    return false;
  }

  bool Int(int i) override { return store_int<int>(i); }
  bool Uint(unsigned u) override { return store_int<unsigned>(u); }
  bool Int64(std::int64_t i) override { return store_int<std::int64_t>(i); }
  bool Uint64(std::uint64_t u) override { return store_int<std::uint64_t>(u); }

  /* Note: This method will only be instantiated (and therefore override the
   * base `bool Double(double)` method) if ElemType is double. */
  bool Double(ElemType d) {
    this->output.push_back(d);
    return true;
  }

  template <typename StringType = ElemType>
  typename std::enable_if<std::is_same<StringType, std::string>::value,
                          bool>::type
  store_string(const char *str, std::size_t length, bool copy) {
    this->output.push_back(ElemType{str, length});
    return true;
  }
  template <typename StringType = ElemType>
  typename std::enable_if<!std::is_same<StringType, std::string>::value,
                          bool>::type
  store_string(const char *, std::size_t, bool) {
    return false;
  }

  bool String(const char *str, std::size_t length, bool copy) override {
    return store_string(str, length, copy);
  }

  bool StartObject(std::true_type) {
    this->output.emplace_back();
    return this->template push_handler<HandlerType, ElemType>(
        this->output.back());
  }

  bool StartObject(std::false_type) { return false; }

  bool StartObject() override {
    return StartObject(
        std::integral_constant<bool, std::is_base_of<OutputHandler<ElemType>,
                                                     HandlerType>::value>{});
  }
};

/*! \brief handler for TreeParam objects from XGBoost schema*/
class TreeParamHandler : public OutputHandler<int> {
 public:
  using OutputHandler<int>::OutputHandler;

  bool String(const char *str, std::size_t length, bool copy) override;
};

/*! \brief handler for RegTree objects from XGBoost schema*/
class RegTreeHandler : public OutputHandler<treelite::Tree<float, float>> {
 public:
  using OutputHandler<treelite::Tree<float, float>>::OutputHandler;
  bool StartArray() override;

  bool StartObject() override;

  bool Uint(unsigned u) override;

  bool EndObject(std::size_t memberCount) override;

 private:
  std::vector<double> loss_changes;
  std::vector<double> sum_hessian;
  std::vector<double> base_weights;
  std::vector<int> left_children;
  std::vector<int> right_children;
  std::vector<int> parents;
  std::vector<int> split_indices;
  std::vector<int> split_type;
  std::vector<int> categories_segments;
  std::vector<int> categories_sizes;
  std::vector<int> categories_nodes;
  std::vector<int> categories;
  std::vector<double> split_conditions;
  std::vector<bool> default_left;
  int num_nodes = 0;
};

/*! \brief handler for GBTreeModel objects from XGBoost schema*/
class GBTreeModelHandler : public OutputHandler<treelite::ModelImpl<float, float>> {
  using OutputHandler<treelite::ModelImpl<float, float>>::OutputHandler;
  bool StartArray() override;
  bool StartObject() override;
};

/*! \brief handler for GradientBoosterHandler objects from XGBoost schema*/
class GradientBoosterHandler : public OutputHandler<treelite::ModelImpl<float, float>> {
 public:
  using OutputHandler<treelite::ModelImpl<float, float>>::OutputHandler;
  bool String(const char *str, std::size_t length, bool copy) override;
  bool StartArray() override;
  bool StartObject() override;
  bool EndObject(std::size_t memberCount) override;
 private:
  std::string name;
  std::vector<double> weight_drop;
};

/*! \brief handler for ObjectiveHandler objects from XGBoost schema*/
class ObjectiveHandler : public OutputHandler<std::string> {
  using OutputHandler<std::string>::OutputHandler;

  bool StartObject() override;

  bool String(const char *str, std::size_t length, bool copy) override;
};

/*! \brief handler for LearnerParam objects from XGBoost schema*/
class LearnerParamHandler : public OutputHandler<treelite::ModelImpl<float, float>> {
 public:
  using OutputHandler<treelite::ModelImpl<float, float>>::OutputHandler;
  bool String(const char *str, std::size_t length, bool copy) override;
};

struct XGBoostModelHandle {
  treelite::ModelImpl<float, float>* model;
  std::vector<unsigned> version;
  std::string objective_name;
};

/*! \brief handler for Learner objects from XGBoost schema*/
class LearnerHandler : public OutputHandler<XGBoostModelHandle> {
 public:
  using OutputHandler<XGBoostModelHandle>::OutputHandler;
  bool StartObject() override;
  bool EndObject(std::size_t memberCount) override;
  bool StartArray() override;

 private:
  std::string objective;
};

/*! \brief handler for XGBoost checkpoint */
class XGBoostCheckpointHandler : public OutputHandler<XGBoostModelHandle> {
 public:
  using OutputHandler<XGBoostModelHandle>::OutputHandler;
  bool StartArray() override;
  bool StartObject() override;
};

/*! \brief handler for XGBoostModel objects from XGBoost schema */
class XGBoostModelHandler : public OutputHandler<XGBoostModelHandle> {
 public:
  using OutputHandler<XGBoostModelHandle>::OutputHandler;
  bool StartArray() override;
  bool StartObject() override;
  bool EndObject(std::size_t memberCount) override;
};

/*! \brief handler for root object of XGBoost schema*/
class RootHandler : public OutputHandler<std::unique_ptr<treelite::Model>> {
 public:
  using OutputHandler<std::unique_ptr<treelite::Model>>::OutputHandler;
  bool StartObject() override;
 private:
  XGBoostModelHandle handle;
};

/*! \brief handler which delegates JSON parsing to stack of delegates*/
class DelegatedHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DelegatedHandler>,
      public Delegator {
 public:
  /*! \brief create DelegatedHandler with empty stack */
  static std::shared_ptr<DelegatedHandler> create_empty() {
    struct make_shared_enabler : public DelegatedHandler {};
    std::shared_ptr<DelegatedHandler> new_handler =
      std::make_shared<make_shared_enabler>();
    return new_handler;
  }

  /*! \brief create DelegatedHandler with initial RootHandler on stack */
  static std::shared_ptr<DelegatedHandler> create() {
    std::shared_ptr<DelegatedHandler> new_handler = create_empty();
    new_handler->push_delegate(std::make_shared<RootHandler>(
      new_handler,
      new_handler->result));
    return new_handler;
  }

  /*! \brief push new handler onto stack, delegating ongoing parsing to it
   *  \param new_delegate the delegate to push onto stack
   */
  void push_delegate(
      std::shared_ptr<BaseHandler> new_delegate) override {
    delegates.push(new_delegate);
  }
  /*! \brief pop handler off of stack, returning parsing responsibility to
   *         previous handler on stack
   */
  void pop_delegate() override {
    delegates.pop();
  }
  std::unique_ptr<treelite::Model> get_result();
  bool Null();
  bool Bool(bool b);
  bool Int(int i);
  bool Uint(unsigned u);
  bool Int64(std::int64_t i);
  bool Uint64(std::uint64_t u);
  bool Double(double d);
  bool String(const char *str, std::size_t length, bool copy);
  bool StartObject();
  bool Key(const char *str, std::size_t length, bool copy);
  bool EndObject(std::size_t memberCount);
  bool StartArray();
  bool EndArray(std::size_t elementCount);

 private:
  DelegatedHandler() : delegates{}, result{treelite::Model::Create<float, float>()} {};

  std::stack<std::shared_ptr<BaseHandler>> delegates;
  std::unique_ptr<treelite::Model> result;
};

}  // namespace details
}  // namespace treelite
#endif  // TREELITE_FRONTEND_XGBOOST_XGBOOST_JSON_H_
