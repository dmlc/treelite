#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <treelite/compiler.h>
#include <iostream>
#include <map>
#include <memory>

using treelite::frontend::TreeBuilder;
using treelite::frontend::ModelBuilder;
using treelite::frontend::Value;
using treelite::TypeInfo;

int main(void) {
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

  std::unique_ptr<treelite::Compiler> compiler{treelite::Compiler::Create("ast_native", "{}")};
  treelite::compiler::CompiledModel cm = compiler->Compile(*model.get());
  for (const auto& kv : cm.files) {
    std::cout << "=================" << kv.first << "=================" << std::endl;
    std::cout << kv.second.content << std::endl;
  }

  return 0;
}
