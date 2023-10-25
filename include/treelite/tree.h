/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file tree.h
 * \brief model structure for tree ensemble
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/contiguous_array.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/pybuffer_frame.h>
#include <treelite/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

/* Indicator that certain functions should be visible from a library (Windows only) */
#if defined(_MSC_VER) || defined(_WIN32)
#define TREELITE_DLL_EXPORT __declspec(dllexport)
#else
#define TREELITE_DLL_EXPORT
#endif

namespace treelite::detail::serializer {

template <typename MixIn>
class Serializer;
template <typename MixIn>
class Deserializer;

}  // namespace treelite::detail::serializer

namespace treelite {

// Used for returning version triple from a Model object
struct Version {
  std::int32_t major_ver;
  std::int32_t minor_ver;
  std::int32_t patch_ver;
};

/*! \brief in-memory representation of a decision tree */
template <typename ThresholdType, typename LeafOutputType>
class Tree {
 public:
  static_assert(std::is_same_v<ThresholdType, float> || std::is_same_v<ThresholdType, double>,
      "ThresholdType must be either float32 or float64");
  static_assert(std::is_same_v<LeafOutputType, float> || std::is_same_v<LeafOutputType, double>,
      "LeafOutputType must be one of uint32_t, float32 or float64");
  static_assert(std::is_same_v<ThresholdType, LeafOutputType>,
      "Unsupported combination of ThresholdType and LeafOutputType");

  Tree() = default;
  ~Tree() = default;
  Tree(Tree const&) = delete;
  Tree& operator=(Tree const&) = delete;
  Tree(Tree&&) noexcept = default;
  Tree& operator=(Tree&&) noexcept = default;

  inline Tree<ThresholdType, LeafOutputType> Clone() const;

 private:
  ContiguousArray<TreeNodeType> node_type_;
  ContiguousArray<std::int32_t> cleft_;
  ContiguousArray<std::int32_t> cright_;
  ContiguousArray<std::int32_t> split_index_;
  ContiguousArray<bool> default_left_;
  ContiguousArray<LeafOutputType> leaf_value_;
  ContiguousArray<ThresholdType> threshold_;
  ContiguousArray<Operator> cmp_;
  ContiguousArray<bool> category_list_right_child_;

  // Leaf vector
  ContiguousArray<LeafOutputType> leaf_vector_;
  ContiguousArray<std::uint64_t> leaf_vector_begin_;
  ContiguousArray<std::uint64_t> leaf_vector_end_;

  // Category list
  ContiguousArray<std::uint32_t> category_list_;
  ContiguousArray<std::uint64_t> category_list_begin_;
  ContiguousArray<std::uint64_t> category_list_end_;

  // Node statistics
  ContiguousArray<std::uint64_t> data_count_;
  ContiguousArray<double> sum_hess_;
  ContiguousArray<double> gain_;
  ContiguousArray<bool> data_count_present_;
  ContiguousArray<bool> sum_hess_present_;
  ContiguousArray<bool> gain_present_;

  bool has_categorical_split_{false};

  /* Note: the following member fields shall be re-computed at serialization time */

  // Number of optional fields in the extension slots
  std::int32_t num_opt_field_per_tree_{0};
  std::int32_t num_opt_field_per_node_{0};

  template <typename WriterType, typename X, typename Y>
  friend void DumpTreeAsJSON(WriterType& writer, Tree<X, Y> const& tree);

  template <typename MixIn>
  friend class detail::serializer::Serializer;
  template <typename MixIn>
  friend class detail::serializer::Deserializer;

 public:
  /*! \brief Number of nodes */
  std::int32_t num_nodes{0};
  /*! \brief Initialize the tree with a single root node */
  inline void Init();
  /*! \brief Allocate a new node and return the node's ID */
  inline int AllocNode();

  /** Getters **/
  /*!
   * \brief Index of the node's left child
   * \param nid ID of node being queried
   */
  inline int LeftChild(int nid) const {
    return cleft_[nid];
  }
  /*!
   * \brief Index of the node's right child
   * \param nid ID of node being queried
   */
  inline int RightChild(int nid) const {
    return cright_[nid];
  }
  /*!
   * \brief Index of the node's "default" child, used when feature is missing
   * \param nid ID of node being queried
   */
  inline int DefaultChild(int nid) const {
    return default_left_[nid] ? cleft_[nid] : cright_[nid];
  }
  /*!
   * \brief Feature index of the node's split condition
   * \param nid ID of node being queried
   */
  inline std::int32_t SplitIndex(int nid) const {
    return split_index_[nid];
  }
  /*!
   * \brief Whether to use the left child node, when the feature in the split condition is missing
   * \param nid ID of node being queried
   */
  inline bool DefaultLeft(int nid) const {
    return default_left_[nid];
  }
  /*!
   * \brief Whether the node is leaf node
   * \param nid ID of node being queried
   */
  inline bool IsLeaf(int nid) const {
    return cleft_[nid] == -1;
  }
  /*!
   * \brief Get leaf value of the leaf node
   * \param nid ID of node being queried
   */
  inline LeafOutputType LeafValue(int nid) const {
    return leaf_value_[nid];
  }
  /*!
   * \brief get leaf vector of the leaf node; useful for multi-class random forest classifier
   * \param nid ID of node being queried
   */
  inline std::vector<LeafOutputType> LeafVector(int nid) const {
    std::size_t const offset_begin = leaf_vector_begin_[nid];
    std::size_t const offset_end = leaf_vector_end_[nid];
    if (offset_begin >= leaf_vector_.Size() || offset_end > leaf_vector_.Size()) {
      // Return empty vector, to indicate the lack of leaf vector
      return std::vector<LeafOutputType>();
    }
    return std::vector<LeafOutputType>(&leaf_vector_[offset_begin], &leaf_vector_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief Tests whether the leaf node has a non-empty leaf vector
   * \param nid ID of node being queried
   */
  inline bool HasLeafVector(int nid) const {
    return leaf_vector_begin_[nid] != leaf_vector_end_[nid];
  }
  /*!
   * \brief Get threshold of the node
   * \param nid ID of node being queried
   */
  inline ThresholdType Threshold(int nid) const {
    return threshold_[nid];
  }
  /*!
   * \brief Get comparison operator
   * \param nid ID of node being queried
   */
  inline Operator ComparisonOp(int nid) const {
    return cmp_[nid];
  }
  /*!
   * \brief Get list of all categories belonging to the left/right child node.
   * See the category_list_right_child_ field of each test node to determine whether this list
   * represents the right child node or the left child node. Categories are integers ranging from 0
   * to (n-1), where n is the number of categories in that particular feature. This list is assumed
   * to be in ascending order.
   *
   * \param nid ID of node being queried
   */
  inline std::vector<std::uint32_t> CategoryList(int nid) const {
    std::size_t const offset_begin = category_list_begin_[nid];
    std::size_t const offset_end = category_list_end_[nid];
    if (offset_begin >= category_list_.Size() || offset_end > category_list_.Size()) {
      // Return empty vector, to indicate the lack of any category list
      // The node might be a numerical split
      return {};
    }
    return std::vector<std::uint32_t>(&category_list_[offset_begin], &category_list_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief Get the type of a node
   * \param nid ID of node being queried
   */
  inline TreeNodeType NodeType(int nid) const {
    return node_type_[nid];
  }
  /*!
   * \brief Test whether this node has data count
   * \param nid ID of node being queried
   */
  inline bool HasDataCount(int nid) const {
    return !data_count_present_.Empty() && data_count_present_[nid];
  }
  /*!
   * \brief Get data count
   * \param nid ID of node being queried
   */
  inline std::uint64_t DataCount(int nid) const {
    return data_count_[nid];
  }

  /*!
   * \brief Test whether this node has hessian sum
   * \param nid ID of node being queried
   */
  inline bool HasSumHess(int nid) const {
    return !sum_hess_present_.Empty() && sum_hess_present_[nid];
  }
  /*!
   * \brief Get hessian sum
   * \param nid ID of node being queried
   */
  inline double SumHess(int nid) const {
    return sum_hess_[nid];
  }
  /*!
   * \brief Test whether this node has gain value
   * \param nid ID of node being queried
   */
  inline bool HasGain(int nid) const {
    return !gain_present_.Empty() && gain_present_[nid];
  }
  /*!
   * \brief Get gain value
   * \param nid ID of node being queried
   */
  inline double Gain(int nid) const {
    return gain_[nid];
  }
  /*!
   * \brief Test whether the list given by CategoryList(nid) is associated with the right
   *        child node or the left child node
   * \param nid ID of node being queried
   */
  inline bool CategoryListRightChild(int nid) const {
    return category_list_right_child_[nid];
  }

  /*!
   * \brief Query whether this tree contains any categorical splits
   */
  inline bool HasCategoricalSplit() const {
    return has_categorical_split_;
  }

  /** Setters **/
  /*!
   * \brief Identify two child nodes of the node
   * \param nid ID of node being modified
   * \param left_child ID of the left child node
   * \param right_child ID of the right child node
   */
  inline void SetChildren(int nid, int left_child, int right_child) {
    cleft_[nid] = left_child;
    cright_[nid] = right_child;
  }
  /*!
   * \brief Create a numerical test
   * \param nid ID of node being updated
   * \param split_index Feature index to split
   * \param threshold Threshold value
   * \param default_left Default direction when feature is unknown
   * \param cmp Comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetNumericalTest(
      int nid, std::int32_t split_index, ThresholdType threshold, bool default_left, Operator cmp);
  /*!
   * \brief Create a categorical test
   * \param nid ID of node being updated
   * \param split_index Feature index to split
   * \param default_left Default direction when feature is unknown
   * \param category_list List of categories to belong to either the right child node or the left
   *                      child node. Set categories_list_right_child parameter to indicate
   *                      which node the category list should represent.
   * \param category_list_right_child Whether category_list indicates the list of categories
   *                                  for the right child node (true) or the left child node
   *                                  (false)
   */
  inline void SetCategoricalTest(int nid, std::int32_t split_index, bool default_left,
      std::vector<std::uint32_t> const& category_list, bool category_list_right_child);
  /*!
   * \brief Set the leaf value of the node
   * \param nid ID of node being updated
   * \param value Leaf value
   */
  inline void SetLeaf(int nid, LeafOutputType value);
  /*!
   * \brief Set the leaf vector of the node; useful for multi-class random forest classifier
   * \param nid ID of node being updated
   * \param leaf_vector Leaf vector
   */
  inline void SetLeafVector(int nid, std::vector<LeafOutputType> const& leaf_vector);
  /*!
   * \brief Set the hessian sum of the node
   * \param nid ID of node being updated
   * \param sum_hess Hessian sum
   */
  inline void SetSumHess(int nid, double sum_hess);
  /*!
   * \brief Set the data count of the node
   * \param nid ID of node being updated
   * \param data_count Data count
   */
  inline void SetDataCount(int nid, std::uint64_t data_count);
  /*!
   * \brief Set the gain value of the node
   * \param nid ID of node being updated
   * \param gain Gain value
   */
  inline void SetGain(int nid, double gain);
};

/*! \brief Typed portion of the model class */
template <typename ThresholdT, typename LeafOutputT>
class ModelPreset {
 public:
  /*! \brief member trees */
  std::vector<Tree<ThresholdT, LeafOutputT>> trees;

  using threshold_type = ThresholdT;
  using leaf_output_type = LeafOutputT;

  /*! \brief disable copy; use default move */
  ModelPreset() = default;
  ~ModelPreset() = default;
  ModelPreset(ModelPreset const&) = delete;
  ModelPreset& operator=(ModelPreset const&) = delete;
  ModelPreset(ModelPreset&&) noexcept = default;
  ModelPreset& operator=(ModelPreset&&) noexcept = default;

  inline TypeInfo GetThresholdType() const {
    return TypeInfoFromType<ThresholdT>();
  }
  inline TypeInfo GetLeafOutputType() const {
    return TypeInfoFromType<LeafOutputT>();
  }
  inline std::size_t GetNumTree() const {
    return trees.size();
  }
  void SetTreeLimit(std::size_t limit) {
    return trees.resize(limit);
  }
};

using ModelPresetVariant = std::variant<ModelPreset<float, float>, ModelPreset<double, double>>;

template <int variant_index>
ModelPresetVariant SetModelPresetVariant(int target_variant_index) {
  ModelPresetVariant result;
  if constexpr (variant_index != std::variant_size_v<ModelPresetVariant>) {
    if (variant_index == target_variant_index) {
      using ModelPresetT = std::variant_alternative_t<variant_index, ModelPresetVariant>;
      result = ModelPresetT();
    } else {
      result = SetModelPresetVariant<variant_index + 1>(target_variant_index);
    }
  }
  return result;
}

/*! \brief Model class for tree ensemble model */
class Model {
 public:
  /*! \brief disable copy; use default move */
  Model()
      : major_ver_(TREELITE_VER_MAJOR),
        minor_ver_(TREELITE_VER_MINOR),
        patch_ver_(TREELITE_VER_PATCH) {}
  virtual ~Model() = default;
  Model(Model const&) = delete;
  Model& operator=(Model const&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  ModelPresetVariant variant_;

  template <typename ThresholdType, typename LeafOutputType>
  inline static std::unique_ptr<Model> Create();
  inline static std::unique_ptr<Model> Create(TypeInfo threshold_type, TypeInfo leaf_output_type);
  inline TypeInfo GetThresholdType() const {
    return std::visit([](auto&& inner) { return inner.GetThresholdType(); }, variant_);
  }
  inline TypeInfo GetLeafOutputType() const {
    return std::visit([](auto&& inner) { return inner.GetLeafOutputType(); }, variant_);
  }

  inline std::size_t GetNumTree() const {
    return std::visit([](auto&& inner) { return inner.GetNumTree(); }, variant_);
  }
  inline void SetTreeLimit(std::size_t limit) {
    std::visit([=](auto&& inner) { return inner.SetTreeLimit(limit); }, variant_);
  }
  void DumpAsJSON(std::ostream& fo, bool pretty_print) const;

  inline std::string DumpAsJSON(bool pretty_print) const {
    std::ostringstream oss;
    DumpAsJSON(oss, pretty_print);
    return oss.str();
  }

  /* Compatibility Matrix:
     +------------------+----------+----------+----------------+-----------+
     |                  | To: =3.9 | To: =4.0 | To: >=4.1,<5.0 | To: >=5.0 |
     +------------------+----------+----------+----------------+-----------+
     | From: =3.9       | Yes      | Yes      | Yes            | No        |
     | From: =4.0       | No       | Yes      | Yes            | Yes       |
     | From: >=4.1,<5.0 | No       | Yes      | Yes            | Yes       |
     | From: >=5.0      | No       | No       | No             | Yes       |
     +------------------+----------+----------+----------------+-----------+ */

  /* In-memory serialization, zero-copy */
  TREELITE_DLL_EXPORT std::vector<PyBufferFrame> SerializeToPyBuffer();
  TREELITE_DLL_EXPORT static std::unique_ptr<Model> DeserializeFromPyBuffer(
      std::vector<PyBufferFrame> const& frames);

  /* Serialization to a file stream */
  void SerializeToStream(std::ostream& os);
  static std::unique_ptr<Model> DeserializeFromStream(std::istream& is);
  /*! \brief Return the Treelite version that produced this Model object. */
  inline Version GetVersion() const {
    return {major_ver_, minor_ver_, patch_ver_};
  }

  /*!
   * \brief Number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  std::int32_t num_feature{0};
  /*! \brief Task type */
  TaskType task_type;
  /*! \brief whether to average tree outputs */
  bool average_tree_output{false};

  /* Task parameters */
  std::int32_t num_target;
  ContiguousArray<std::int32_t> num_class;
  ContiguousArray<std::int32_t> leaf_vector_shape;
  /* Per-tree metadata */
  ContiguousArray<std::int32_t> target_id;
  ContiguousArray<std::int32_t> class_id;
  /* Other model parameters */
  std::string postprocessor;
  float sigmoid_alpha{1.0f};
  float ratio_c{1.0f};
  ContiguousArray<double> base_scores;
  std::string attributes;

 private:
  /* Note: the following member fields shall be re-computed at serialization time */
  // Number of trees
  std::uint64_t num_tree_{0};
  // Number of optional fields in the extension slot
  std::int32_t num_opt_field_per_model_{0};
  // Which Treelite version produced this model
  std::int32_t major_ver_;
  std::int32_t minor_ver_;
  std::int32_t patch_ver_;
  // Type parameters
  TypeInfo threshold_type_{TypeInfo::kInvalid};
  TypeInfo leaf_output_type_{TypeInfo::kInvalid};

  template <typename MixIn>
  friend class detail::serializer::Serializer;
  template <typename MixIn>
  friend class detail::serializer::Deserializer;
};

/*!
 * \brief Concatenate multiple model objects into a single model object by copying
 *        all member trees into the destination model object
 * \param objs List of model objects
 * \return Concatenated model
 */
std::unique_ptr<Model> ConcatenateModelObjects(std::vector<Model const*> const& objs);

}  // namespace treelite

#include <treelite/detail/tree.h>

#endif  // TREELITE_TREE_H_
