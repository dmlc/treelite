/*!
 * Copyright (c) 2021 by Contributors
 * \file ast_native.h
 * \brief C code generator
 * \author Hyunsu Cho
 */

#ifndef TREELITE_COMPILER_AST_NATIVE_H_
#define TREELITE_COMPILER_AST_NATIVE_H_

#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <memory>

namespace treelite {
namespace compiler {

class ASTNativeCompilerImpl;

class ASTNativeCompiler : public Compiler {
 public:
  explicit ASTNativeCompiler(const CompilerParam& param);
  virtual ~ASTNativeCompiler();
  CompiledModel Compile(const Model& model) override;
 private:
  std::unique_ptr<ASTNativeCompilerImpl> pimpl_;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_NATIVE_H_
