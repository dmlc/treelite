#include "./builder.h"
#include "ast.pb.h"
#include <cmath>

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(serialize);

void ASTBuilder::Serialize(std::ostream* output) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  treelite_ast_protobuf::ASTTree ast;
  treelite_ast_protobuf::ASTNode* head = ast.mutable_head();
  this->main_node->Serialize(head);
  ast.set_num_feature(this->num_feature);
  ast.set_num_output_group(this->num_output_group);
  ast.set_random_forest_flag(this->random_forest_flag);
  ast.set_output_vector_flag(this->output_vector_flag);
  ast.set_quantize_threshold_flag(this->quantize_threshold_flag);
  ast.mutable_model_param()->insert(this->model_param.begin(),
                                    this->model_param.end());
  ast.SerializeToOstream(output);
}

void ASTBuilder::Serialize(const std::string& filename) {
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(filename.c_str(), "w"));
  dmlc::ostream os(fo.get());
  this->Serialize(&os);
}

}  // namespace compiler
}  // namespace treelite
