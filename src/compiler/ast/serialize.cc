#include "./builder.h"
#include <cmath>

#ifdef TREELITE_PROTOBUF_SUPPORT

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "ast.pb.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(serialize);

void ASTBuilder::Serialize(std::ostream* output, bool binary) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  treelite_ast_protobuf::ASTTree ast;
  treelite_ast_protobuf::ASTNode* head = ast.mutable_head();
  this->main_node->Serialize(head);
  ast.set_num_feature(this->num_feature);
  ast.set_num_output_group(this->num_output_group);
  ast.set_random_forest_flag(this->random_forest_flag);
  ast.set_output_vector_flag(this->output_vector_flag);
  ast.set_quantize_threshold_flag(this->quantize_threshold_flag);
  google::protobuf::RepeatedField<bool>
    is_categorical_pf(this->is_categorical.begin(), this->is_categorical.end());
  ast.mutable_is_categorical()->Swap(&is_categorical_pf);
  ast.mutable_model_param()->insert(this->model_param.begin(),
                                    this->model_param.end());
  if (binary) {
    ast.SerializeToOstream(output);
  } else {
    google::protobuf::io::OstreamOutputStream os(output);
    google::protobuf::TextFormat::Print(ast, &os);
  }
}

void ASTBuilder::Serialize(const std::string& filename, bool binary) {
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(filename.c_str(), "w"));
  dmlc::ostream os(fo.get());
  this->Serialize(&os, binary);
}

}  // namespace compiler
}  // namespace treelite

#else   // TREELITE_PROTOBUF_SUPPORT

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(serialize);

void ASTBuilder::Serialize(std::ostream* output, bool binary) {
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
}

void ASTBuilder::Serialize(const std::string& filename, bool binary) {
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
}

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_PROTOBUF_SUPPORT
