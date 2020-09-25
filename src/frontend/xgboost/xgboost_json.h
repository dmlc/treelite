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

class Delegator {
 public:
  virtual void pop_delegate() = 0;
  virtual void push_delegate(std::shared_ptr<BaseHandler> new_delegate) = 0;
};

class BaseHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, BaseHandler> {
 public:
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
  template <typename HandlerType, typename... ArgsTypes>
  bool push_handler(ArgsTypes &... args) {
    if (auto parent = BaseHandler::delegator.lock()) {
      parent->push_delegate(std::make_shared<HandlerType>(delegator, args...));
      return true;
    } else {
      return false;
    }
  }

  template <typename HandlerType, typename... ArgsTypes>
  bool push_key_handler(std::string key, ArgsTypes &... args) {
    if (check_cur_key(key)) {
      push_handler<HandlerType, ArgsTypes...>(args...);
      return true;
    } else {
      return false;
    }
  }
  bool pop_handler();
  void set_cur_key(const char *str, std::size_t length);
  const std::string &get_cur_key();
  bool check_cur_key(const std::string &query_key);
  template <typename ValueType>
  bool assign_value(const std::string &key,
                    ValueType &&value,
                    ValueType &output);
  template <typename ValueType>
  bool assign_value(const std::string &key,
                    const ValueType &value,
                    ValueType &output);

 private:
  std::weak_ptr<Delegator> delegator;
  std::string cur_key;
};

class IgnoreHandler : public BaseHandler {
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

template <typename OutputType> class OutputHandler : public BaseHandler {
 public:
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &output)
      : BaseHandler{parent_delegator}, m_output{output} {};
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &&output) = delete;

 protected:
  OutputType &m_output;
};

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

class TreeParamHandler : public OutputHandler<int> {
 public:
  using OutputHandler<int>::OutputHandler;

  bool String(const char *str, std::size_t length, bool copy);
};

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

class GBTreeModelHandler : public OutputHandler<treelite::Model> {
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray();
  bool StartObject();
};

class GradientBoosterHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool Uint(unsigned u);
  bool String(const char *str, std::size_t length, bool copy);
  bool StartObject();
};

class ObjectiveHandler : public OutputHandler<std::string> {
  using OutputHandler<std::string>::OutputHandler;

  bool StartObject();

  bool String(const char *str, std::size_t length, bool copy);
};

class LearnerParamHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool String(const char *str, std::size_t length, bool copy);
};

class LearnerHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray();
  bool StartObject();
  bool EndObject(std::size_t memberCount);

 private:
  std::string objective;
};

class XGBoostModelHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray();
  bool StartObject();
  bool EndObject(std::size_t memberCount);

 private:
  std::vector<unsigned> version;
};

class RootHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartObject();
};

class DelegatedHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DelegatedHandler>,
      public Delegator {

 public:
  static std::shared_ptr<DelegatedHandler> create() {
    struct make_shared_enabler : public DelegatedHandler {};

    std::shared_ptr<DelegatedHandler> new_handler = \
      std::make_shared<make_shared_enabler>();
    new_handler->push_delegate(std::make_shared<RootHandler>(
      new_handler,
      new_handler->result));
    return new_handler;
  }

  void push_delegate(
      std::shared_ptr<BaseHandler> new_delegate) override {
    delegates.push(new_delegate);
  }
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
