/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file builder.cc
 * \brief Explicit template specializations for the ASTBuilder class
 * \author Hyunsu Cho
 */

#include "./builder.h"

namespace treelite {
namespace compiler {

// Explicit template specializations
// (https://docs.microsoft.com/en-us/cpp/cpp/source-code-organization-cpp-templates)
template class ASTBuilder<float, uint32_t>;
template class ASTBuilder<float, float>;
template class ASTBuilder<double, uint32_t>;
template class ASTBuilder<double, double>;

}  // namespace compiler
}  // namespace treelite
