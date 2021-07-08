/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file compiler.h
 * \brief Interface of compiler that compiles a tree ensemble model
 * \author Hyunsu Cho
 */
#ifndef TREELITE_COMPILER_H_
#define TREELITE_COMPILER_H_

#include <unordered_map>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace treelite {

class Model;  // forward declaration

namespace compiler {

struct CompilerParam;  // forward declaration

struct CompiledModel {
  struct FileEntry {
    std::string content;
    std::vector<char> content_binary;
    bool is_binary;
    FileEntry() : is_binary(false) {}
    // Passing std::vector<char> indicates binary data
    // Passing std::string indicates text data
    // Use move constructor and assignment exclusively to save memory
    explicit FileEntry(const std::string& content) = delete;
    explicit FileEntry(std::string&& content)
      : content(std::move(content)), is_binary(false) {}
    explicit FileEntry(const std::vector<char>&) = delete;
    explicit FileEntry(std::vector<char>&& content)
      : content_binary(std::move(content)), is_binary(true) {}
    FileEntry(const FileEntry& other) = delete;
    FileEntry(FileEntry&& other) = default;
    FileEntry& operator=(const FileEntry& other) = delete;
    FileEntry& operator=(FileEntry&& other) = default;
  };
  std::string backend;
  std::unordered_map<std::string, FileEntry> files;
  std::string file_prefix;
};

}  // namespace compiler

/*! \brief interface of compiler */
class Compiler {
 public:
  /*! \brief virtual destructor */
  virtual ~Compiler() = default;
  /*!
   * \brief convert tree ensemble model
   * \return compiled model
   */
  virtual compiler::CompiledModel Compile(const Model& model) = 0;
  /*!
   * \brief Query the parameters used to intiailize the compiler
   * \return parameters used
   */
  virtual compiler::CompilerParam QueryParam() const = 0;
  /*!
   * \brief create a compiler from given name
   * \param name name of compiler
   * \param param_json_string JSON string representing
   * \return The created compiler
   */
  static Compiler* Create(const std::string& name,
                          const char* param_json_str);
};

}  // namespace treelite

#endif  // TREELITE_COMPILER_H_
