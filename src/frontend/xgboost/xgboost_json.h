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
  virtual bool Bool(bool b) { return false; }
  virtual bool Int(int i) { return false; }
  virtual bool Uint(unsigned u) { return false; }
  virtual bool Int64(int64_t i) { return false; }
  virtual bool Uint64(uint64_t u) { return false; }
  virtual bool Double(double d) { return false; }
  virtual bool String(const char *str, std::size_t length, bool copy) {
    return false;
  }
  virtual bool StartObject() { return false; }
  virtual bool Key(const char *str, std::size_t length, bool copy) {
    set_cur_key(str, length);
    return true;
  }
  virtual bool EndObject(std::size_t memberCount) { return pop_handler(); }
  virtual bool StartArray() { return false; }
  virtual bool EndArray(std::size_t elementCount) { return pop_handler(); }

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
  bool Null();
  bool Bool(bool b);
  bool Int(int i);
  bool Uint(unsigned u);
  bool Int64(int64_t i);
  bool Uint64(uint64_t u);
  bool Double(double d);
  bool String(const char *str, std::size_t length, bool copy);
  bool StartObject();
  bool Key(const char *str, std::size_t length, bool copy);
  bool StartArray();
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
                OutputType &output)
      : BaseHandler{parent_delegator}, m_output{output} {};
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &&output) = delete;

 protected:
  /* \brief the output value constructed or modified during parsing */
  OutputType &m_output;
};

/*! \brief handler for array of objects of given type*/
template <typename ElemType, typename HandlerType = BaseHandler>
class ArrayHandler : public OutputHandler<std::vector<ElemType>> {
 public:
  using OutputHandler<std::vector<ElemType>>::OutputHandler;
  bool Bool(ElemType b) {
    this->m_output.push_back(b);
    return true;
  }
  template <typename ArgType, typename IntType = ElemType>
  typename std::enable_if<std::is_integral<IntType>::value, bool>::type
  store_int(ArgType i) {
    this->m_output.push_back(static_cast<ElemType>(i));
    return true;
  }

  template <typename ArgType, typename IntType = ElemType>
  typename std::enable_if<!std::is_integral<IntType>::value, bool>::type
  store_int(ArgType i) {
    return false;
  }

  bool Int(int i) { return store_int<int>(i); }
  bool Uint(unsigned u) { return store_int<unsigned>(u); }
  bool Int64(int64_t i) { return store_int<int64_t>(i); }
  bool Uint64(uint64_t u) { return store_int<uint64_t>(u); }
  bool Double(ElemType d) {
    this->m_output.push_back(d);
    return true;
  }

  template <typename StringType = ElemType>
  typename std::enable_if<std::is_same<StringType, std::string>::value,
                          bool>::type
  store_string(const char *str, std::size_t length, bool copy) {
    this->m_output.push_back(ElemType{str, length});
    return true;
  }
  template <typename StringType = ElemType>
  typename std::enable_if<!std::is_same<StringType, std::string>::value,
                          bool>::type
  store_string(const char *str, std::size_t length, bool copy) {
    return false;
  }

  bool String(const char *str, std::size_t length, bool copy) {
    return store_string(str, length, copy);
  }

  bool StartObject(std::true_type) {
    this->m_output.emplace_back();
    return this->template push_handler<HandlerType, ElemType>(
        this->m_output.back());
  }

  bool StartObject(std::false_type) { return false; }

  bool StartObject() {
    return StartObject(
        std::integral_constant<bool, std::is_base_of<OutputHandler<ElemType>,
                                                     HandlerType>::value>{});
  }
};

/*! \brief handler for TreeParam objects from XGBoost schema*/
class TreeParamHandler : public OutputHandler<int> {
 public:
  using OutputHandler<int>::OutputHandler;

  bool String(const char *str, std::size_t length, bool copy);
};

/*! \brief handler for RegTree objects from XGBoost schema*/
class RegTreeHandler : public OutputHandler<treelite::Tree> {
 public:
  using OutputHandler<treelite::Tree>::OutputHandler;
  bool StartArray();

  bool StartObject();

  bool Uint(unsigned u);

  bool EndObject(std::size_t memberCount);

 private:
  std::vector<double> loss_changes;
  std::vector<double> sum_hessian;
  std::vector<double> base_weights;
  std::vector<int> leaf_child_counts;
  std::vector<int> left_children;
  std::vector<int> right_children;
  std::vector<int> parents;
  std::vector<int> split_indices;
  std::vector<double> split_conditions;
  std::vector<bool> default_left;
  int num_nodes;
};

/*! \brief handler for GBTreeModel objects from XGBoost schema*/
class GBTreeModelHandler : public OutputHandler<treelite::Model> {
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray();
  bool StartObject();
};

/*! \brief handler for GradientBoosterHandler objects from XGBoost schema*/
class GradientBoosterHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool Uint(unsigned u);
  bool String(const char *str, std::size_t length, bool copy);
  bool StartObject();
};

/*! \brief handler for ObjectiveHandler objects from XGBoost schema*/
class ObjectiveHandler : public OutputHandler<std::string> {
  using OutputHandler<std::string>::OutputHandler;

  bool StartObject();

  bool String(const char *str, std::size_t length, bool copy);
};

/*! \brief handler for LearnerParam objects from XGBoost schema*/
class LearnerParamHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool String(const char *str, std::size_t length, bool copy);
};

/*! \brief handler for Learner objects from XGBoost schema*/
class LearnerHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray();
  bool StartObject();
  bool EndObject(std::size_t memberCount);

 private:
  std::string objective;
};

/*! \brief handler for XGBoostModel objects from XGBoost schema*/
class XGBoostModelHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray();
  bool StartObject();
  bool EndObject(std::size_t memberCount);

 private:
  std::vector<unsigned> version;
};

/*! \brief handler for root object of XGBoost schema*/
class RootHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartObject();
};

/*! \brief handler which delegates JSON parsing to stack of delegates*/
class DelegatedHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DelegatedHandler>,
      public Delegator {

 public:
  /*! \brief create DelegatedHandler with initial RootHandler on stack */
  static std::shared_ptr<DelegatedHandler> create() {
    struct make_shared_enabler : public DelegatedHandler {};

    std::shared_ptr<DelegatedHandler> new_handler = \
      std::make_shared<make_shared_enabler>();
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
  treelite::Model get_result();
  bool Null();
  bool Bool(bool b);
  bool Int(int i);
  bool Uint(unsigned u);
  bool Int64(int64_t i);
  bool Uint64(uint64_t u);
  bool Double(double d);
  bool String(const char *str, std::size_t length, bool copy);
  bool StartObject();
  bool Key(const char *str, std::size_t length, bool copy);
  bool EndObject(std::size_t memberCount);
  bool StartArray();
  bool EndArray(std::size_t elementCount);

 private:
  DelegatedHandler() : delegates{}, result{} {};

  std::stack<std::shared_ptr<BaseHandler>> delegates;
  treelite::Model result;
};

}  // namespace details
}  // namespace treelite
#endif  // TREELITE_FRONTEND_XGBOOST_XGBOOST_JSON_H_
