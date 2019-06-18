/*!
 * Copyright 2017 by Contributors
 * \file compiler.h
 * \brief Interface of compiler that compiles a tree ensemble model
 * \author Philip Cho
 */
#ifndef TREELITE_COMPILER_H_
#define TREELITE_COMPILER_H_

#include <dmlc/registry.h>
#include <treelite/common.h>
#include <unordered_map>
#include <functional>
#include <memory>
#include <string>

namespace treelite {

struct Model;  // forward declaration

namespace compiler {

struct CompilerParam;  // forward declaration

struct CompiledModel {
  std::string backend;
  std::unordered_map<std::string, std::string> files;
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
   * \brief create a compiler from given name
   * \param name name of compiler
   * \return The created compiler
   */
  static Compiler* Create(const std::string& name,
                          const compiler::CompilerParam& param);
};

/*!
 * \brief Registry entry for compiler
 */
struct CompilerReg
    : public dmlc::FunctionRegEntryBase<CompilerReg,
                  std::function<Compiler* (const compiler::CompilerParam&)> > {
};

/*!
 * \brief Macro to register compiler.
 *
 * \code
 * // example of registering the simple compiler
 * TREELITE_REGISTER_COMPILER(SimpleCompiler, "simple")
 * .describe("Bare-bones simple compiler")
 * .set_body([]() {
 *     return new SimpleCompiler();
 *   });
 * \endcode
 */
#define TREELITE_REGISTER_COMPILER(UniqueId, Name)                            \
  static DMLC_ATTRIBUTE_UNUSED ::treelite::CompilerReg &          \
  __make_ ## CompilerReg ## _ ## UniqueId ## __ =                \
      ::dmlc::Registry< ::treelite::CompilerReg>::Get()->__REGISTER__(Name)

}  // namespace treelite

#endif  // TREELITE_COMPILER_H_
