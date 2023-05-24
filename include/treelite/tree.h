/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file tree.h
 * \brief model structure for tree ensemble
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/base.h>
#include <treelite/contiguous_array.h>
#include <treelite/logging.h>
#include <treelite/task_type.h>
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
#include <vector>

#define __TREELITE_STR(x) #x
#define _TREELITE_STR(x) __TREELITE_STR(x)

#define TREELITE_MAX_PRED_TRANSFORM_LENGTH 256

/* Indicator that certain functions should be visible from a library (Windows only) */
#if defined(_MSC_VER) || defined(_WIN32)
#define TREELITE_DLL_EXPORT __declspec(dllexport)
#else
#define TREELITE_DLL_EXPORT
#endif

namespace treelite {

class GTILBridge;

template <typename ThresholdType, typename LeafOutputType>
class ModelImpl;

// Used for returning version triple from a Model object
struct Version {
  std::int32_t major_ver;
  std::int32_t minor_ver;
  std::int32_t patch_ver;
};

// Represent a frame in the Python buffer protocol (PEP 3118). We use a simplified representation
// to hold only 1-D arrays with stride 1.
struct PyBufferFrame {
  void* buf;
  char* format;
  std::size_t itemsize;
  std::size_t nitem;
};

static_assert(std::is_pod<PyBufferFrame>::value, "PyBufferFrame must be a POD type");

/*! \brief Group of parameters that are dependent on the choice of the task type. */
struct TaskParam {
  enum class OutputType : uint8_t { kFloat = 0, kInt = 1 };
  /*! \brief The type of output from each leaf node. */
  OutputType output_type;
  /*!
   * \brief Whether we designate a subset of the trees to compute the prediction for each class.
   *
   * If True, the prediction for the i-th class is determined by the trees whose index is congruent
   * to [i] modulo [num_class]. Only applicable if we are performing classification task with
   * num_class > 2.
   */
  bool grove_per_class;
  /*!
   * \brief The number of classes in the target label.
   *
   * The num_class field should be >1 only when we're performing multi-class classification.
   * Otherwise, for tasks such as binary classification, regression, and learning-to-rank, set
   * num_class=1.
   */
  unsigned int num_class;
  /*!
   * \brief Dimension of the output from each leaf node.
   *
   * If >1, each leaf node produces a 1D vector output. If =1, each leaf node produces a single
   * scalar.
   */
  unsigned int leaf_vector_size;
};

inline std::string OutputTypeToString(TaskParam::OutputType type) {
  switch (type) {
  case TaskParam::OutputType::kFloat:
    return "float";
  case TaskParam::OutputType::kInt:
    return "int";
  default:
    return "";
  }
}

inline TaskParam::OutputType StringToOutputType(std::string const& str) {
  if (str == "float") {
    return TaskParam::OutputType::kFloat;
  } else if (str == "int") {
    return TaskParam::OutputType::kInt;
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized output type: " << str;
    return TaskParam::OutputType::kFloat;  // to avoid compiler warning
  }
}

static_assert(std::is_pod<TaskParam>::value, "TaskParameter must be POD type");

/*! \brief in-memory representation of a decision tree */
template <typename ThresholdType, typename LeafOutputType>
class Tree {
 public:
  /*! \brief tree node */
  struct Node {
    /*! \brief Initialization method.
     * Use this in lieu of constructor (POD types cannot have a non-trivial constructor) */
    inline void Init();
    /*! \brief store either leaf value or decision threshold */
    union Info {
      LeafOutputType leaf_value;  // for leaf nodes
      ThresholdType threshold;  // for non-leaf nodes
    };
    /*! \brief pointer to left and right children */
    std::int32_t cleft_, cright_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    std::uint32_t sindex_;
    /*! \brief storage for leaf value or decision threshold */
    Info info_;
    /*!
     * \brief number of data points whose traversal paths include this node.
     *        LightGBM models natively store this statistics.
     */
    std::uint64_t data_count_;
    /*!
     * \brief sum of hessian values for all data points whose traversal paths
     *        include this node. This value is generally correlated positively
     *        with the data count. XGBoost models natively store this
     *        statistics.
     */
    double sum_hess_;
    /*!
     * \brief change in loss that is attributed to a particular split
     */
    double gain_;
    /*! \brief feature split type */
    SplitFeatureType split_type_;
    /*!
     * \brief operator to use for expression of form [fval] OP [threshold].
     * If the expression evaluates to true, take the left child;
     * otherwise, take the right child.
     */
    Operator cmp_;
    /*! \brief whether data_count_ field is present */
    bool data_count_present_;
    /*! \brief whether sum_hess_ field is present */
    bool sum_hess_present_;
    /*! \brief whether gain_present_ field is present */
    bool gain_present_;
    /* \brief whether the list given by MatchingCategories(nid) is associated with the right child
     *        node or the left child node. True if the right child, False otherwise */
    bool categories_list_right_child_;

    /** Getters **/
    inline int LeftChild() const {
      return cleft_;
    }
    inline int RightChild() const {
      return cright_;
    }
    inline bool DefaultLeft() const {
      // Extract the most significant bit (MSB) of sindex_, which encodes the default_left field
      return (sindex_ >> 31U) != 0;
    }
    inline int DefaultChild() const {
      // Extract the most significant bit (MSB) of sindex_, which encodes the default_left field
      return ((sindex_ >> 31U) != 0) ? cleft_ : cright_;
    }
    inline std::uint32_t SplitIndex() const {
      // Extract all bits except the most significant bit (MSB) from sindex_.
      return (sindex_ & ((1U << 31U) - 1U));
    }
    inline bool IsLeaf() const {
      return cleft_ == -1;
    }
    inline LeafOutputType LeafValue() const {
      return info_.leaf_value;
    }
    inline ThresholdType Threshold() const {
      return info_.threshold;
    }
    inline Operator ComparisonOp() const {
      return cmp_;
    }
    inline SplitFeatureType SplitType() const {
      return split_type_;
    }
    inline bool HasDataCount() const {
      return data_count_present_;
    }
    inline std::uint64_t DataCount() const {
      return data_count_;
    }
    inline bool HasSumHess() const {
      return sum_hess_present_;
    }
    inline double SumHess() const {
      return sum_hess_;
    }
    inline bool HasGain() const {
      return gain_present_;
    }
    inline double Gain() const {
      return gain_;
    }
    inline bool CategoriesListRightChild() const {
      return categories_list_right_child_;
    }
  };

  static_assert(std::is_pod<Node>::value, "Node must be a POD type");
  static_assert(
      std::is_same<ThresholdType, float>::value || std::is_same<ThresholdType, double>::value,
      "ThresholdType must be either float32 or float64");
  static_assert(std::is_same<LeafOutputType, uint32_t>::value
                    || std::is_same<LeafOutputType, float>::value
                    || std::is_same<LeafOutputType, double>::value,
      "LeafOutputType must be one of uint32_t, float32 or float64");
  static_assert(std::is_same<ThresholdType, LeafOutputType>::value
                    || std::is_same<LeafOutputType, uint32_t>::value,
      "Unsupported combination of ThresholdType and LeafOutputType");
  static_assert((std::is_same<ThresholdType, float>::value && sizeof(Node) == 48)
                    || (std::is_same<ThresholdType, double>::value && sizeof(Node) == 56),
      "Node size incorrect");

  explicit Tree(bool use_opt_field = true);
  ~Tree() = default;
  Tree(Tree const&) = delete;
  Tree& operator=(Tree const&) = delete;
  Tree(Tree&&) noexcept = default;
  Tree& operator=(Tree&&) noexcept = default;

  inline Tree<ThresholdType, LeafOutputType> Clone() const;

  inline char const* GetFormatStringForNode();
  inline void GetPyBuffer(std::vector<PyBufferFrame>* dest);
  inline void SerializeToStream(std::ostream& os);
  // Load a Tree object from a sequence of PyBuffer frames
  // Returns the updated position of the cursor in the sequence
  inline std::vector<PyBufferFrame>::iterator InitFromPyBuffer(
      std::vector<PyBufferFrame>::iterator it);
  inline void DeserializeFromStream(std::istream& is);

 private:
  // vector of nodes
  ContiguousArray<Node> nodes_;
  ContiguousArray<LeafOutputType> leaf_vector_;
  // Map nid to the start and end index in leaf_vector_
  // We could use std::pair, but it is not POD, so easier to use two vectors
  // here
  ContiguousArray<std::size_t> leaf_vector_begin_;
  ContiguousArray<std::size_t> leaf_vector_end_;
  ContiguousArray<std::uint32_t> matching_categories_;
  ContiguousArray<std::size_t> matching_categories_offset_;
  bool has_categorical_split_{false};

  /* Note: the following member fields shall be re-computed at serialization time */
  // Whether to use optional fields
  bool use_opt_field_{false};
  // Number of optional fields in the extension slots
  int32_t num_opt_field_per_tree_{0};
  int32_t num_opt_field_per_node_{0};

  template <typename WriterType, typename X, typename Y>
  friend void DumpModelAsJSON(WriterType& writer, ModelImpl<X, Y> const& model);
  template <typename WriterType, typename X, typename Y>
  friend void DumpTreeAsJSON(WriterType& writer, Tree<X, Y> const& tree);

  // allocate a new node
  inline int AllocNode();

  // utility functions used for serialization, internal use only
  template <typename ScalarHandler, typename PrimitiveArrayHandler, typename CompositeArrayHandler>
  inline void SerializeTemplate(ScalarHandler scalar_handler,
      PrimitiveArrayHandler primitive_array_handler, CompositeArrayHandler composite_array_handler);
  template <typename ScalarHandler, typename ArrayHandler, typename SkipOptFieldHandlerFunc>
  inline void DeserializeTemplate(ScalarHandler scalar_handler, ArrayHandler array_handler,
      SkipOptFieldHandlerFunc skip_opt_field_handler);

  friend class GTILBridge;  // bridge to enable optimized access to nodes from GTIL

 public:
  /*! \brief number of nodes */
  int num_nodes{0};
  /*! \brief initialize the model with a single root node */
  inline void Init();
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid);

  /** Getters **/
  /*!
   * \brief index of the node's left child
   * \param nid ID of node being queried
   */
  inline int LeftChild(int nid) const {
    return nodes_[nid].LeftChild();
  }
  /*!
   * \brief index of the node's right child
   * \param nid ID of node being queried
   */
  inline int RightChild(int nid) const {
    return nodes_[nid].RightChild();
  }
  /*!
   * \brief index of the node's "default" child, used when feature is missing
   * \param nid ID of node being queried
   */
  inline int DefaultChild(int nid) const {
    return nodes_[nid].DefaultChild();
  }
  /*!
   * \brief feature index of the node's split condition
   * \param nid ID of node being queried
   */
  inline std::uint32_t SplitIndex(int nid) const {
    return nodes_[nid].SplitIndex();
  }
  /*!
   * \brief whether to use the left child node, when the feature in the split condition is missing
   * \param nid ID of node being queried
   */
  inline bool DefaultLeft(int nid) const {
    return nodes_[nid].DefaultLeft();
  }
  /*!
   * \brief whether the node is leaf node
   * \param nid ID of node being queried
   */
  inline bool IsLeaf(int nid) const {
    return nodes_[nid].IsLeaf();
  }
  /*!
   * \brief get leaf value of the leaf node
   * \param nid ID of node being queried
   */
  inline LeafOutputType LeafValue(int nid) const {
    return nodes_[nid].LeafValue();
  }
  /*!
   * \brief get leaf vector of the leaf node; useful for multi-class random forest classifier
   * \param nid ID of node being queried
   */
  inline std::vector<LeafOutputType> LeafVector(int nid) const {
    const std::size_t offset_begin = leaf_vector_begin_[nid];
    const std::size_t offset_end = leaf_vector_end_[nid];
    if (offset_begin >= leaf_vector_.Size() || offset_end > leaf_vector_.Size()) {
      // Return empty vector, to indicate the lack of leaf vector
      return std::vector<LeafOutputType>();
    }
    return std::vector<LeafOutputType>(&leaf_vector_[offset_begin], &leaf_vector_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief tests whether the leaf node has a non-empty leaf vector
   * \param nid ID of node being queried
   */
  inline bool HasLeafVector(int nid) const {
    return leaf_vector_begin_[nid] != leaf_vector_end_[nid];
  }
  /*!
   * \brief get threshold of the node
   * \param nid ID of node being queried
   */
  inline ThresholdType Threshold(int nid) const {
    return nodes_[nid].Threshold();
  }
  /*!
   * \brief get comparison operator
   * \param nid ID of node being queried
   */
  inline Operator ComparisonOp(int nid) const {
    return nodes_[nid].ComparisonOp();
  }
  /*!
   * \brief Get list of all categories belonging to the left/right child node. See the
   *        categories_list_right_child_ field of each split to determine whether this list
   * represents the right child node or the left child node. Categories are integers ranging from 0
   * to (n-1), where n is the number of categories in that particular feature. This list is assumed
   * to be in ascending order. \param nid ID of node being queried
   */
  inline std::vector<std::uint32_t> MatchingCategories(int nid) const {
    const std::size_t offset_begin = matching_categories_offset_[nid];
    const std::size_t offset_end = matching_categories_offset_[nid + 1];
    if (offset_begin >= matching_categories_.Size() || offset_end > matching_categories_.Size()) {
      // Return empty vector, to indicate the lack of any matching categories
      // The node might be a numerical split
      return std::vector<std::uint32_t>();
    }
    return std::vector<std::uint32_t>(
        &matching_categories_[offset_begin], &matching_categories_[offset_end]);
    // Use unsafe access here, since we may need to take the address of one past the last
    // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief get feature split type
   * \param nid ID of node being queried
   */
  inline SplitFeatureType SplitType(int nid) const {
    return nodes_[nid].SplitType();
  }
  /*!
   * \brief test whether this node has data count
   * \param nid ID of node being queried
   */
  inline bool HasDataCount(int nid) const {
    return nodes_[nid].HasDataCount();
  }
  /*!
   * \brief get data count
   * \param nid ID of node being queried
   */
  inline std::uint64_t DataCount(int nid) const {
    return nodes_[nid].DataCount();
  }

  /*!
   * \brief test whether this node has hessian sum
   * \param nid ID of node being queried
   */
  inline bool HasSumHess(int nid) const {
    return nodes_[nid].HasSumHess();
  }
  /*!
   * \brief get hessian sum
   * \param nid ID of node being queried
   */
  inline double SumHess(int nid) const {
    return nodes_[nid].SumHess();
  }
  /*!
   * \brief test whether this node has gain value
   * \param nid ID of node being queried
   */
  inline bool HasGain(int nid) const {
    return nodes_[nid].HasGain();
  }
  /*!
   * \brief get gain value
   * \param nid ID of node being queried
   */
  inline double Gain(int nid) const {
    return nodes_[nid].Gain();
  }
  /*!
   * \brief test whether the list given by MatchingCategories(nid) is associated with the right
   *        child node or the left child node
   * \param nid ID of node being queried
   */
  inline bool CategoriesListRightChild(int nid) const {
    return nodes_[nid].CategoriesListRightChild();
  }

  /*!
   * \brief Query whether this tree contains any categorical splits
   */
  inline bool HasCategoricalSplit() const {
    return has_categorical_split_;
  }

  /** Setters **/
  /*!
   * \brief create a numerical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param threshold threshold value
   * \param default_left the default direction when feature is unknown
   * \param cmp comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetNumericalSplit(
      int nid, unsigned split_index, ThresholdType threshold, bool default_left, Operator cmp);
  /*!
   * \brief create a categorical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param default_left the default direction when feature is unknown
   * \param categories_list list of categories to belong to either the right child node or the left
   *                        child node. Set categories_list_right_child parameter to indicate
   *                        which node the category list should represent.
   * \param categories_list_right_child whether categories_list indicates the list of categories
   *                                    for the right child node (true) or the left child node
   *                                    (false)
   */
  inline void SetCategoricalSplit(int nid, unsigned split_index, bool default_left,
      std::vector<uint32_t> const& categories_list, bool categories_list_right_child);
  /*!
   * \brief set the leaf value of the node
   * \param nid ID of node being updated
   * \param value leaf value
   */
  inline void SetLeaf(int nid, LeafOutputType value);
  /*!
   * \brief set the leaf vector of the node; useful for multi-class random forest classifier
   * \param nid ID of node being updated
   * \param leaf_vector leaf vector
   */
  inline void SetLeafVector(int nid, std::vector<LeafOutputType> const& leaf_vector);
  /*!
   * \brief set the hessian sum of the node
   * \param nid ID of node being updated
   * \param sum_hess hessian sum
   */
  inline void SetSumHess(int nid, double sum_hess) {
    Node& node = nodes_.at(nid);
    node.sum_hess_ = sum_hess;
    node.sum_hess_present_ = true;
  }
  /*!
   * \brief set the data count of the node
   * \param nid ID of node being updated
   * \param data_count data count
   */
  inline void SetDataCount(int nid, uint64_t data_count) {
    Node& node = nodes_.at(nid);
    node.data_count_ = data_count;
    node.data_count_present_ = true;
  }
  /*!
   * \brief set the gain value of the node
   * \param nid ID of node being updated
   * \param gain gain value
   */
  inline void SetGain(int nid, double gain) {
    Node& node = nodes_.at(nid);
    node.gain_ = gain;
    node.gain_present_ = true;
  }
};

struct ModelParam {
  /*!
   * \defgroup model_param Extra parameters for tree ensemble models
   * \{
   */
  /*!
   * \brief name of prediction transform function
   *
   * This parameter specifies how to transform raw margin values into
   * final predictions. By default, this is set to `'identity'`, which
   * means no transformation.
   *
   * For the **multi-class classification task**, `pred_transfrom` must be one
   * of the following values:
   * \snippet src/compiler/pred_transform.cc pred_transform_multiclass_db
   *
   * For **all other tasks** (e.g. regression, binary classification, ranking
   * etc.), `pred_transfrom` must be one of the following values:
   * \snippet src/compiler/pred_transform.cc pred_transform_db
   *
   */
  char pred_transform[TREELITE_MAX_PRED_TRANSFORM_LENGTH] = {0};
  /*!
   * \brief scaling parameter for sigmoid function
   * `sigmoid(x) = 1 / (1 + exp(-alpha * x))`
   *
   * This parameter is used only when `pred_transform` is set to `'sigmoid'`.
   * It must be strictly positive; if unspecified, it is set to 1.0.
   */
  float sigmoid_alpha;
  /*!
   * \brief scaling parameter for exponential standard ratio transformation
   * `expstdratio(x) = exp2(-x / c)`
   *
   * This parameter is used only when `pred_transform` is set to `'exponential_standard_ratio'`.
   * If unspecified, it is set to 1.0.
   */
  float ratio_c;
  /*!
   * \brief global bias of the model
   *
   * Predicted margin scores of all instances will be adjusted by the global
   * bias. If unspecified, the bias is set to zero.
   */
  float global_bias;
  /*! \} */

  ModelParam() : sigmoid_alpha(1.0f), ratio_c(1.0f), global_bias(0.0f) {
    std::memset(pred_transform, 0, TREELITE_MAX_PRED_TRANSFORM_LENGTH * sizeof(char));
    std::strncpy(pred_transform, "identity", sizeof(pred_transform));
  }
  ~ModelParam() = default;
  ModelParam(ModelParam const&) = default;
  ModelParam& operator=(ModelParam const&) = default;
  ModelParam(ModelParam&&) = default;
  ModelParam& operator=(ModelParam&&) = default;

  template <typename Container>
  inline std::vector<std::pair<std::string, std::string>> InitAllowUnknown(Container const& kwargs);
  inline std::map<std::string, std::string> __DICT__() const;
};

static_assert(
    std::is_standard_layout<ModelParam>::value, "ModelParam must be in the standard layout");

inline void InitParamAndCheck(
    ModelParam* param, std::vector<std::pair<std::string, std::string>> const& cfg);

/*! \brief thin wrapper for tree ensemble model */
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

  template <typename ThresholdType, typename LeafOutputType>
  inline static std::unique_ptr<Model> Create();
  inline static std::unique_ptr<Model> Create(TypeInfo threshold_type, TypeInfo leaf_output_type);
  inline TypeInfo GetThresholdType() const {
    return threshold_type_;
  }
  inline TypeInfo GetLeafOutputType() const {
    return leaf_output_type_;
  }
  template <typename Func>
  inline auto Dispatch(Func func);
  template <typename Func>
  inline auto Dispatch(Func func) const;

  virtual std::size_t GetNumTree() const = 0;
  virtual void SetTreeLimit(std::size_t limit) = 0;
  virtual void DumpAsJSON(std::ostream& fo, bool pretty_print) const = 0;

  inline std::string DumpAsJSON(bool pretty_print) const {
    std::ostringstream oss;
    DumpAsJSON(oss, pretty_print);
    return oss.str();
  }

  /* Compatibility Matrix:
     +------------------+----------+----------+----------------+-----------+
     |                  | To: =2.4 | To: =3.0 | To: >=3.1,<4.0 | To: >=4.0 |
     +------------------+----------+----------+----------------+-----------+
     | From: <2.4       | No       | No       | No             | No        |
     | From: =2.4       | Yes      | Yes      | Yes            | No        |
     | From: =3.0       | No       | Yes      | Yes            | Yes       |
     | From: >=3.1,<4.0 | No       | Yes      | Yes            | Yes       |
     | From: >=4.0      | No       | No       | No             | Yes       |
     +------------------+----------+----------+----------------+-----------+ */

  /* In-memory serialization, zero-copy */
  TREELITE_DLL_EXPORT std::vector<PyBufferFrame> GetPyBuffer();
  TREELITE_DLL_EXPORT static std::unique_ptr<Model> CreateFromPyBuffer(
      std::vector<PyBufferFrame> frames);

  /* Serialization to a file stream */
  void SerializeToStream(std::ostream& os);
  static std::unique_ptr<Model> DeserializeFromStream(std::istream& is);
  /*! \brief Return the Treelite version that produced this Model object. */
  inline Version GetVersion() const {
    return {major_ver_, minor_ver_, patch_ver_};
  }

  /*!
   * \brief number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  int32_t num_feature{0};
  /*! \brief Task type */
  TaskType task_type;
  /*! \brief whether to average tree outputs */
  bool average_tree_output{false};
  /*! \brief Group of parameters that are specific to the particular task type */
  TaskParam task_param{};
  /*! \brief extra parameters */
  ModelParam param{};

 protected:
  /* Note: the following member fields shall be re-computed at serialization time */
  // Number of trees
  uint64_t num_tree_{0};
  // Number of optional fields in the extension slot
  int32_t num_opt_field_per_model_{0};
  // Which Treelite version produced this model
  int32_t major_ver_;
  int32_t minor_ver_;
  int32_t patch_ver_;

 private:
  TypeInfo threshold_type_{TypeInfo::kInvalid};
  TypeInfo leaf_output_type_{TypeInfo::kInvalid};
  // Internal functions for serialization
  virtual void GetPyBuffer(std::vector<PyBufferFrame>* dest) = 0;
  virtual void SerializeToStreamImpl(std::ostream& os) = 0;
  // Load a Model object from a sequence of PyBuffer frames
  // Returns the updated position of the cursor in the sequence
  virtual std::vector<PyBufferFrame>::iterator InitFromPyBuffer(
      std::vector<PyBufferFrame>::iterator it, std::size_t num_frame)
      = 0;
  virtual void DeserializeFromStreamImpl(std::istream& is) = 0;
  template <typename HeaderPrimitiveFieldHandlerFunc>
  inline void SerializeTemplate(HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler);
  template <typename HeaderPrimitiveFieldHandlerFunc>
  inline static void DeserializeTemplate(
      HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler, int32_t& major_ver,
      int32_t& minor_ver, int32_t& patch_ver, TypeInfo& threshold_type, TypeInfo& leaf_output_type);
};

template <typename ThresholdType, typename LeafOutputType>
class ModelImpl : public Model {
 public:
  /*! \brief member trees */
  std::vector<Tree<ThresholdType, LeafOutputType>> trees;

  /*! \brief disable copy; use default move */
  ModelImpl() = default;
  ~ModelImpl() override = default;
  ModelImpl(ModelImpl const&) = delete;
  ModelImpl& operator=(ModelImpl const&) = delete;
  ModelImpl(ModelImpl&&) noexcept = default;
  ModelImpl& operator=(ModelImpl&&) noexcept = default;

  void DumpAsJSON(std::ostream& fo, bool pretty_print) const override;
  inline std::size_t GetNumTree() const override {
    return trees.size();
  }
  void SetTreeLimit(std::size_t limit) override {
    return trees.resize(limit);
  }

  inline void GetPyBuffer(std::vector<PyBufferFrame>* dest) override;
  inline void SerializeToStreamImpl(std::ostream& os) override;
  // Load a ModelImpl object from a sequence of PyBuffer frames
  // Returns the updated position of the cursor in the sequence
  inline std::vector<PyBufferFrame>::iterator InitFromPyBuffer(
      std::vector<PyBufferFrame>::iterator it, std::size_t num_frame) override;
  inline void DeserializeFromStreamImpl(std::istream& is) override;

 private:
  template <typename HeaderPrimitiveFieldHandlerFunc, typename HeaderCompositeFieldHandlerFunc,
      typename TreeHandlerFunc>
  inline void SerializeTemplate(HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler,
      HeaderCompositeFieldHandlerFunc header_composite_field_handler, TreeHandlerFunc tree_handler);
  template <typename HeaderFieldHandlerFunc, typename TreeHandlerFunc,
      typename SkipOptFieldHandlerFunc>
  inline void DeserializeTemplate(size_t num_tree, HeaderFieldHandlerFunc header_field_handler,
      TreeHandlerFunc tree_handler, SkipOptFieldHandlerFunc skip_opt_field_handler);
};

/*!
 * \brief Concatenate multiple model objects into a single model object by copying
 *        all member trees into the destination model object
 * \param objs List of model objects
 * \return Concatenated model
 */
std::unique_ptr<Model> ConcatenateModelObjects(std::vector<Model const*> const& objs);

}  // namespace treelite

#include "tree_impl.h"

#endif  // TREELITE_TREE_H_
