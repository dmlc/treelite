#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <treelite/compiler.h>
#include <treelite/compiler_param.h>
#include <iostream>
#include <map>
#include <memory>

using treelite::frontend::TreeBuilder;
using treelite::frontend::ModelBuilder;

int main(void) {
  std::unique_ptr<TreeBuilder> tree{new TreeBuilder};
  tree->CreateNode(0);
  tree->CreateNode(1);
  tree->CreateNode(2);
  tree->SetNumericalTestNode(0, 0, "<", 0.0f, true, 1, 2);
  tree->SetLeafNode(1, -1.0f);
  tree->SetLeafNode(2, 1.0f);
  tree->SetRootNode(0);

  std::unique_ptr<ModelBuilder> builder{new ModelBuilder(2, 1, false)};
  builder->InsertTree(tree.get());

  std::unique_ptr<treelite::Model> model = builder->CommitModel();
  std::cout << model->GetNumTree() << std::endl;

  treelite::compiler::CompilerParam param;
  param.Init(std::map<std::string, std::string>{});
  std::unique_ptr<treelite::Compiler> compiler{treelite::Compiler::Create("ast_native", param)};
  treelite::compiler::CompiledModel cm = compiler->Compile(*model.get());
  for (const auto& kv : cm.files) {
    std::cout << "=================" << kv.first << "=================" << std::endl;
    std::cout << kv.second.content << std::endl;
  }

  return 0;
}
