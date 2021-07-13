/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file json_serializer.cc
 * \brief Reference serializer implementation, which serializes to JSON. This is useful for testing
 *        correctness of the binary serializer
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <treelite/logging.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <ostream>
#include <cstdint>
#include <cstddef>

namespace {

template <typename WriterType>
void WriteElement(WriterType& writer, double e) {
  writer.Double(e);
}

template <typename WriterType, typename ThresholdType, typename LeafOutputType>
void WriteNode(WriterType& writer,
               const typename treelite::Tree<ThresholdType, LeafOutputType>::Node& node) {
  writer.StartObject();

  writer.Key("cleft");
  writer.Int(node.cleft_);
  writer.Key("cright");
  writer.Int(node.cright_);
  writer.Key("split_index");
  writer.Uint(node.sindex_ & ((1U << 31U) - 1U));
  writer.Key("default_left");
  writer.Bool((node.sindex_ >> 31U) != 0);
  if (node.cleft_ == -1) {
    writer.Key("leaf_value");
    writer.Double(node.info_.leaf_value);
  } else {
    writer.Key("threshold");
    writer.Double(node.info_.threshold);
  }
  if (node.data_count_present_) {
    writer.Key("data_count");
    writer.Uint64(node.data_count_);
  }
  if (node.sum_hess_present_) {
    writer.Key("sum_hess");
    writer.Double(node.sum_hess_);
  }
  if (node.gain_present_) {
    writer.Key("gain");
    writer.Double(node.gain_);
  }
  writer.Key("split_type");
  writer.Int(static_cast<int8_t>(node.split_type_));
  writer.Key("cmp");
  writer.Int(static_cast<int8_t>(node.cmp_));
  writer.Key("categories_list_right_child");
  writer.Bool(node.categories_list_right_child_);

  writer.EndObject();
}

template <typename WriterType, typename ElementType>
void WriteContiguousArray(WriterType& writer,
                          const treelite::ContiguousArray<ElementType>& array) {
  writer.StartArray();
  for (std::size_t i = 0; i < array.Size(); ++i) {
    WriteElement(writer, array[i]);
  }
  writer.EndArray();
}

template <typename WriterType>
void SerializeTaskParamToJSON(WriterType& writer, treelite::TaskParam task_param) {
  writer.StartObject();

  writer.Key("output_type");
  writer.Uint(static_cast<uint8_t>(task_param.output_type));
  writer.Key("grove_per_class");
  writer.Bool(task_param.grove_per_class);
  writer.Key("num_class");
  writer.Uint(task_param.num_class);
  writer.Key("leaf_vector_size");
  writer.Uint(task_param.leaf_vector_size);

  writer.EndObject();
}

template <typename WriterType>
void SerializeModelParamToJSON(WriterType& writer, treelite::ModelParam model_param) {
  writer.StartObject();

  writer.Key("pred_transform");
  std::string pred_transform(model_param.pred_transform);
  writer.String(pred_transform.data(), pred_transform.size());
  writer.Key("sigmoid_alpha");
  writer.Double(model_param.sigmoid_alpha);
  writer.Key("global_bias");
  writer.Double(model_param.global_bias);

  writer.EndObject();
}

}  // anonymous namespace

namespace treelite {

template <typename WriterType, typename ThresholdType, typename LeafOutputType>
void DumpTreeAsJSON(WriterType& writer, const Tree<ThresholdType, LeafOutputType>& tree) {
  writer.StartObject();

  writer.Key("num_nodes");
  writer.Int(tree.num_nodes);
  writer.Key("leaf_vector");
  WriteContiguousArray(writer, tree.leaf_vector_);
  writer.Key("leaf_vector_offset");
  WriteContiguousArray(writer, tree.leaf_vector_offset_);
  writer.Key("matching_categories");
  WriteContiguousArray(writer, tree.matching_categories_);
  writer.Key("matching_categories_offset");
  WriteContiguousArray(writer, tree.matching_categories_offset_);
  writer.Key("nodes");
  writer.StartArray();
  for (std::size_t i = 0; i < tree.nodes_.Size(); ++i) {
    WriteNode<WriterType, ThresholdType, LeafOutputType>(writer, tree.nodes_[i]);
  }
  writer.EndArray();

  writer.EndObject();

  // Sanity check
  TREELITE_CHECK_EQ(tree.nodes_.Size(), tree.num_nodes);
  TREELITE_CHECK_EQ(tree.nodes_.Size() + 1, tree.leaf_vector_offset_.Size());
  TREELITE_CHECK_EQ(tree.leaf_vector_offset_.Back(), tree.leaf_vector_.Size());
  TREELITE_CHECK_EQ(tree.nodes_.Size() + 1, tree.matching_categories_offset_.Size());
  TREELITE_CHECK_EQ(tree.matching_categories_offset_.Back(), tree.matching_categories_.Size());
}

template <typename WriterType, typename ThresholdType, typename LeafOutputType>
void DumpModelAsJSON(WriterType& writer,
                     const ModelImpl<ThresholdType, LeafOutputType>& model) {
  writer.StartObject();

  writer.Key("num_feature");
  writer.Int(model.num_feature);
  writer.Key("task_type");
  writer.Uint(static_cast<uint8_t>(model.task_type));
  writer.Key("average_tree_output");
  writer.Bool(model.average_tree_output);
  writer.Key("task_param");
  SerializeTaskParamToJSON(writer, model.task_param);
  writer.Key("model_param");
  SerializeModelParamToJSON(writer, model.param);
  writer.Key("trees");
  writer.StartArray();
  for (const Tree<ThresholdType, LeafOutputType>& tree : model.trees) {
    DumpTreeAsJSON(writer, tree);
  }
  writer.EndArray();

  writer.EndObject();
}

template <typename ThresholdType, typename LeafOutputType>
void
ModelImpl<ThresholdType, LeafOutputType>::DumpAsJSON(std::ostream& fo, bool pretty_print) const {
  rapidjson::OStreamWrapper os(fo);
  if (pretty_print) {
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(os);
    writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
    DumpModelAsJSON(writer, *this);
  } else {
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(os);
    DumpModelAsJSON(writer, *this);
  }
}

template void ModelImpl<float, uint32_t>::DumpAsJSON(std::ostream& fo, bool pretty_print) const;
template void ModelImpl<float, float>::DumpAsJSON(std::ostream& fo, bool pretty_print) const;
template void ModelImpl<double, uint32_t>::DumpAsJSON(std::ostream& fo, bool pretty_print) const;
template void ModelImpl<double, double>::DumpAsJSON(std::ostream& fo, bool pretty_print) const;

}  // namespace treelite
