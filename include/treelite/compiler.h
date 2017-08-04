/*!
 * Copyright 2017 by Contributors
 * \file compiler.h
 * \brief Interface of compiler that translates a tree ensemble model into
 *        a semantic model
 * \author Philip Cho
 */
#ifndef TREELITE_COMPILER_H_
#define TREELITE_COMPILER_H_

#include <dmlc/registry.h>
#include <treelite/common.h>
#include <functional>
#include <memory>

namespace treelite {

struct Model;  // forward declaration

namespace compiler {
  struct CompilerParam;  // forward declaration
}  // namespace compiler

namespace semantic {
class CodeBlock;  // forward declaration
}  // namespace semantic

/*!
 * \brief interface of compiler
 */
class Compiler {
 public:
  /*! \brief virtual destructor */
  virtual ~Compiler() = default;
  /*!
   * \brief convert tree ensemble model into semantic model
   * \return semantic model
   */
  virtual std::unique_ptr<semantic::CodeBlock>
  Export(const Model& model) = 0;
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
