/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file compiler.cc
 * \brief Registry of compilers
 */
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include "./ast_native.h"
#include "./failsafe.h"

namespace treelite {
Compiler* Compiler::Create(const std::string& name,
                           const compiler::CompilerParam& param) {
  if (name == "ast_native") {
    return new compiler::ASTNativeCompiler(param);
  } else if (name == "failsafe") {
    return new compiler::FailSafeCompiler(param);
  } else {
    LOG(FATAL) << "Unrecognized compiler '" << name << "'";
    return nullptr;
  }
}
}  // namespace treelite

namespace treelite {
namespace compiler {
// register compiler parameter
DMLC_REGISTER_PARAMETER(CompilerParam);

}  // namespace compiler
}  // namespace treelite
