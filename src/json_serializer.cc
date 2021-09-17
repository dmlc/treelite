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
#include <type_traits>
#include <cstdint>
#include <cstddef>

namespace {

template <typename WriterType, typename T,
    typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Uint64(static_cast<uint64_t>(e));
}

template <typename WriterType, typename T,
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Double(static_cast<double>(e));
}

template <typename WriterType>
void WriteString(WriterType& writer, const std::string& str) {
  writer.String(str.data(), str.size());
}

template <typename WriterType, typename ThresholdType, typename LeafOutputType>
void WriteNode(WriterType& writer,
               const treelite::Tree<ThresholdType, LeafOutputType>& tree,
               int node_id) {
  writer.StartObject();

  writer.Key("node_id");
  writer.Int(node_id);
  if (tree.IsLeaf(node_id)) {
    writer.Key("leaf_value");
    if (tree.HasLeafVector(node_id)) {
      writer.StartArray();
      for (LeafOutputType e : tree.LeafVector(node_id)) {
        WriteElement(writer, e);
      }
      writer.EndArray();
    } else {
      WriteElement(writer, tree.LeafValue(node_id));
    }
  } else {
    writer.Key("split_feature_id");
    writer.Uint(tree.SplitIndex(node_id));
    writer.Key("default_left");
    writer.Bool(tree.DefaultLeft(node_id));
    writer.Key("split_type");
    auto split_type = tree.SplitType(node_id);
    WriteString(writer, treelite::SplitFeatureTypeName(split_type));
    if (split_type == treelite::SplitFeatureType::kNumerical) {
      writer.Key("comparison_op");
      WriteString(writer, treelite::OpName(tree.ComparisonOp(node_id)));
      writer.Key("threshold");
      writer.Double(tree.Threshold(node_id));
    } else if (split_type == treelite::SplitFeatureType::kCategorical) {
      writer.Key("categories_list_right_child");
      writer.Bool(tree.CategoriesListRightChild(node_id));
      TREELITE_CHECK(tree.HasMatchingCategories(node_id))
        << "Test node with categorical test must have a list of matching categories";
      writer.Key("matching_categories");
      writer.StartArray();
      for (uint32_t e : tree.MatchingCategories(node_id)) {
        writer.Uint(e);
      }
      writer.EndArray();
    }
    writer.Key("left_child");
    writer.Int(tree.LeftChild(node_id));
    writer.Key("right_child");
    writer.Int(tree.RightChild(node_id));
  }
  if (tree.HasDataCount(node_id)) {
    writer.Key("data_count");
    writer.Uint64(tree.DataCount(node_id));
  }
  if (tree.HasSumHess(node_id)) {
    writer.Key("sum_hess");
    writer.Double(tree.SumHess(node_id));
  }
  if (tree.HasGain(node_id)) {
    writer.Key("gain");
    writer.Double(tree.Gain(node_id));
  }

  writer.EndObject();
}

template <typename WriterType>
void SerializeTaskParamToJSON(WriterType& writer, treelite::TaskParam task_param) {
  writer.StartObject();

  writer.Key("output_type");
  WriteString(writer, treelite::OutputTypeToString(task_param.output_type));
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
  WriteString(writer, std::string(model_param.pred_transform));
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
  writer.Key("nodes");
  writer.StartArray();
  for (std::size_t i = 0; i < tree.nodes_.Size(); ++i) {
    WriteNode<WriterType, ThresholdType, LeafOutputType>(writer, tree, i);
  }
  writer.EndArray();

  writer.EndObject();

  // Basic checks
  TREELITE_CHECK_EQ(tree.nodes_.Size(), tree.num_nodes);
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
  WriteString(writer, TaskTypeToString(model.task_type));
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
