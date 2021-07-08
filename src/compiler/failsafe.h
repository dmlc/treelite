/*!
 * Copyright (c) 2021 by Contributors
 * \file failsafe.h
 * \brief C code generator (fail-safe). The generated code will mimic prediction logic found in
 *        XGBoost
 * \author Hyunsu Cho
 */

#ifndef TREELITE_COMPILER_FAILSAFE_H_
#define TREELITE_COMPILER_FAILSAFE_H_

#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <memory>

namespace treelite {
namespace compiler {

class FailSafeCompilerImpl;

class FailSafeCompiler : public Compiler {
 public:
  explicit FailSafeCompiler(const CompilerParam& param);
  CompiledModel Compile(const Model& model) override;
  CompilerParam QueryParam() const override;
 private:
  std::unique_ptr<FailSafeCompilerImpl> pimpl_;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_FAILSAFE_H_
