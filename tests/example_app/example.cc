/*!
 * Copyright (c) 2023 by Contributors
 * \file example.cc
 * \brief Test using Treelite as a C++ library
 */
#include <treelite/c_api.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

#include <iostream>
#include <map>
#include <memory>

using treelite::TypeInfo;
using treelite::frontend::ModelBuilder;
using treelite::frontend::TreeBuilder;
using treelite::frontend::Value;

int main(void) {
  std::cout << "TREELITE_VERSION = " << TREELITE_VERSION << std::endl;
  auto tree = std::make_unique<TreeBuilder>(TypeInfo::kFloat32, TypeInfo::kFloat32);
  tree->CreateNode(0);
  tree->CreateNode(1);
  tree->CreateNode(2);
  tree->SetNumericalTestNode(0, 0, "<", Value::Create<float>(0), true, 1, 2);
  tree->SetLeafNode(1, Value::Create<float>(-1.0));
  tree->SetLeafNode(2, Value::Create<float>(1.0));
  tree->SetRootNode(0);

  auto builder
      = std::make_unique<ModelBuilder>(2, 1, false, TypeInfo::kFloat32, TypeInfo::kFloat32);
  builder->InsertTree(tree.get());

  auto model = builder->CommitModel();
  std::cout << model->GetNumTree() << std::endl;
  return 0;
}
