/*!
 * Copyright (c) 2020-2023 by Contributors
 * \file json_serializer.cc
 * \brief JSON serializer for Tree objects. This is useful for inspecting the content of
 *        Tree objects. Note: we do not provide a deserializer from JSON.
 * \author Hyunsu Cho
 */

#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>
#include <treelite/contiguous_array.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <type_traits>

namespace {

template <typename WriterType, typename T,
    typename std::enable_if<std::is_same_v<T, std::uint64_t>, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Uint64(e);
}

template <typename WriterType, typename T,
    typename std::enable_if<std::is_same_v<T, std::uint32_t>, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Uint(e);
}

template <typename WriterType, typename T,
    typename std::enable_if<std::is_same_v<T, std::int64_t>, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Int64(e);
}

template <typename WriterType, typename T,
    typename std::enable_if<std::is_same_v<T, std::int32_t>, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Int(e);
}

template <typename WriterType, typename T,
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
void WriteElement(WriterType& writer, T e) {
  writer.Double(static_cast<double>(e));
}

template <typename WriterType, typename T>
void WriteArray(WriterType& writer, treelite::ContiguousArray<T> const& array) {
  writer.StartArray();
  for (std::size_t i = 0; i < array.Size(); ++i) {
    WriteElement(writer, array[i]);
  }
  writer.EndArray();
}

template <typename WriterType, typename T>
void WriteArray(WriterType& writer, std::vector<T> const& array) {
  writer.StartArray();
  for (auto const& e : array) {
    WriteElement(writer, e);
  }
  writer.EndArray();
}

template <typename WriterType>
void WriteString(WriterType& writer, std::string const& str) {
  writer.String(str.data(), str.size());
}

template <typename WriterType, typename ThresholdType, typename LeafOutputType>
void WriteNode(
    WriterType& writer, treelite::Tree<ThresholdType, LeafOutputType> const& tree, int node_id) {
  writer.StartObject();

  writer.Key("node_id");
  writer.Int(node_id);
  if (tree.IsLeaf(node_id)) {
    writer.Key("leaf_value");
    if (tree.HasLeafVector(node_id)) {
      WriteArray(writer, tree.LeafVector(node_id));
    } else {
      WriteElement(writer, tree.LeafValue(node_id));
    }
  } else {
    writer.Key("split_feature_id");
    writer.Uint(tree.SplitIndex(node_id));
    writer.Key("default_left");
    writer.Bool(tree.DefaultLeft(node_id));
    writer.Key("node_type");
    auto node_type = tree.NodeType(node_id);
    WriteString(writer, treelite::TreeNodeTypeToString(node_type));
    if (node_type == treelite::TreeNodeType::kNumericalTestNode) {
      writer.Key("comparison_op");
      WriteString(writer, treelite::OperatorToString(tree.ComparisonOp(node_id)));
      writer.Key("threshold");
      writer.Double(tree.Threshold(node_id));
    } else if (node_type == treelite::TreeNodeType::kCategoricalTestNode) {
      writer.Key("category_list_right_child");
      writer.Bool(tree.CategoryListRightChild(node_id));
      writer.Key("category_list");
      WriteArray(writer, tree.CategoryList(node_id));
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
void SerializeTaskParametersToJSON(WriterType& writer, treelite::Model const& model) {
  writer.Key("num_target");
  writer.Int(model.num_target);
  writer.Key("num_class");
  WriteArray(writer, model.num_class);
  writer.Key("leaf_vector_shape");
  WriteArray(writer, model.leaf_vector_shape);
}

template <typename WriterType>
void SerializeModelParametersToJSON(WriterType& writer, treelite::Model const& model) {
  writer.Key("postprocessor");
  WriteString(writer, model.postprocessor);
  writer.Key("sigmoid_alpha");
  writer.Double(model.sigmoid_alpha);
  writer.Key("ratio_c");
  writer.Double(model.ratio_c);
  writer.Key("base_scores");
  WriteArray(writer, model.base_scores);
  writer.Key("attributes");
  WriteString(writer, model.attributes);
}

}  // anonymous namespace

namespace treelite {

template <typename WriterType, typename ThresholdType, typename LeafOutputType>
void DumpTreeAsJSON(WriterType& writer, Tree<ThresholdType, LeafOutputType> const& tree) {
  writer.StartObject();

  writer.Key("num_nodes");
  writer.Int(tree.num_nodes);
  writer.Key("has_categorical_split");
  writer.Bool(tree.has_categorical_split_);
  writer.Key("nodes");
  writer.StartArray();
  for (std::size_t i = 0; i < tree.num_nodes; ++i) {
    WriteNode<WriterType, ThresholdType, LeafOutputType>(writer, tree, i);
  }
  writer.EndArray();

  writer.EndObject();
}

template <typename WriterType>
void DumpModelAsJSON(WriterType& writer, Model const& model) {
  writer.StartObject();

  writer.Key("threshold_type");
  WriteString(writer, TypeInfoToString(model.GetThresholdType()));
  writer.Key("leaf_output_type");
  WriteString(writer, TypeInfoToString(model.GetLeafOutputType()));
  writer.Key("num_feature");
  writer.Int(model.num_feature);
  writer.Key("task_type");
  WriteString(writer, TaskTypeToString(model.task_type));
  writer.Key("average_tree_output");
  writer.Bool(model.average_tree_output);

  SerializeTaskParametersToJSON(writer, model);

  writer.Key("target_id");
  WriteArray(writer, model.target_id);
  writer.Key("class_id");
  WriteArray(writer, model.class_id);

  SerializeModelParametersToJSON(writer, model);

  writer.Key("trees");
  writer.StartArray();
  std::visit(
      [&writer](auto&& concrete_model) {
        for (auto const& tree : concrete_model.trees) {
          DumpTreeAsJSON(writer, tree);
        }
      },
      model.variant_);
  writer.EndArray();

  writer.EndObject();
}

void Model::DumpAsJSON(std::ostream& fo, bool pretty_print) const {
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

}  // namespace treelite
