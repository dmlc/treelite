/*!
 * Copyright (c) 2023 by Contributors
 * \file example.cc
 * \brief Test using Treelite as a C++ library
 */
#include <treelite/c_api.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <iostream>
#include <memory>
#include <vector>

int main(void) {
  std::cout << "TREELITE_VERSION = " << TREELITE_VERSION << std::endl;
  auto builder = treelite::model_builder::GetModelBuilder(treelite::TypeInfo::kFloat32,
      treelite::TypeInfo::kFloat32,
      treelite::model_builder::Metadata{2, treelite::TaskType::kRegressor, false, 1, {1}, {1, 1}},
      treelite::model_builder::TreeAnnotation{1, {0}, {0}},
      treelite::model_builder::PostProcessorFunc{"identity"}, std::vector<double>{0.0});
  builder->StartTree();
  builder->StartNode(0);
  builder->NumericalTest(0, 0.0, true, treelite::Operator::kLT, 1, 2);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafScalar(-1.0);
  builder->EndNode();
  builder->StartNode(2);
  builder->LeafScalar(1.0);
  builder->EndNode();
  builder->EndTree();

  auto model = builder->CommitModel();
  std::cout << model->GetNumTree() << std::endl;
  return 0;
}
