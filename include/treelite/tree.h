/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file tree.h
 * \brief model structure for tree ensemble
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/base.h>
#include <treelite/version.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>

#define __TREELITE_STR(x) #x
#define _TREELITE_STR(x) __TREELITE_STR(x)

#define TREELITE_MAX_PRED_TRANSFORM_LENGTH 256

/* Foward declarations */
namespace dmlc {

class Stream;
float stof(const std::string& value, std::size_t* pos);

}  // namespace dmlc

namespace treelite {

// Represent a frame in the Python buffer protocol (PEP 3118). We use a simplified representation
// to hold only 1-D arrays with stride 1.
struct PyBufferFrame {
  void* buf;
  char* format;
  std::size_t itemsize;
  std::size_t nitem;
};

// Serialize a frame to a file stream
void SerializePyBufferFrame(PyBufferFrame frame, FILE* dest_fp);

// Deserialize a frame from a file stream
// Note. This function allocates new buffers for buf and format fields and returns the references
// via the last two arguments. Make sure to free them to avoid memory leak.
PyBufferFrame DeserializePyBufferFrame(
    FILE* src_fp, void** allocated_buf, char** allocated_format);

static_assert(std::is_pod<PyBufferFrame>::value, "PyBufferFrame must be a POD type");

template <typename T>
class ContiguousArray {
 public:
  ContiguousArray();
  ~ContiguousArray();
  // NOTE: use Clone to make deep copy; copy constructors disabled
  ContiguousArray(const ContiguousArray&) = delete;
  ContiguousArray& operator=(const ContiguousArray&) = delete;
  ContiguousArray(ContiguousArray&& other) noexcept;
  ContiguousArray& operator=(ContiguousArray&& other) noexcept;
  inline ContiguousArray Clone() const;
  inline void UseForeignBuffer(void* prealloc_buf, std::size_t size, bool assume_ownership);
    // Set assume_ownership=true to transfer the ownership of the buffer to the ContiguousArray
    // object. The object will be responsible for freeing the buffer.
  inline T* Data();
  inline const T* Data() const;
  inline T* End();
  inline const T* End() const;
  inline T& Back();
  inline const T& Back() const;
  inline std::size_t Size() const;
  inline void Reserve(std::size_t newsize);
  inline void Resize(std::size_t newsize);
  inline void Resize(std::size_t newsize, T t);
  inline void Clear();
  inline void PushBack(T t);
  inline void Extend(const std::vector<T>& other);
  /* Unsafe access, no bounds checking */
  inline T& operator[](std::size_t idx);
  inline const T& operator[](std::size_t idx) const;
  /* Safe access, with bounds checking */
  inline T& at(std::size_t idx);
  inline const T& at(std::size_t idx) const;
  /* Safe access, with bounds checking + check against non-existent node (<0) */
  inline T& at(int idx);
  inline const T& at(int idx) const;
  static_assert(std::is_pod<T>::value, "T must be POD");

 private:
  T* buffer_;
  std::size_t size_;
  std::size_t capacity_;
  bool owned_buffer_;
};

/*!
 * \brief Enum type representing the task type.
 *
 * The task type places constraints on the parameters of TaskParameter. See the docstring for each
 * enum constants for more details.
 */
enum class TaskType : uint8_t {
  /*!
   * \brief Catch-all task type encoding all tasks that are not multi-class classification, such as
   *        binary classification, regression, and learning-to-rank.
   *
   * The kBinaryClfRegr task type implies the following constraints on the task parameters:
   * output_type=float, grove_per_class=false, num_class=1, leaf_vector_size=1.
   */
  kBinaryClfRegr = 0,
  /*!
   * \brief The multi-class classification task, in which the prediction for each class is given
   *        by the sum of outputs from a subset of the trees. We refer to this method as
   *        "grove-per-class".
   *
   * In this setting, each leaf node in a tree produces a single scalar output. To obtain
   * predictions for each class, we divide the trees into multiple groups ("groves") and then
   * compute the sum of outputs of the trees in each group. The prediction for the i-th class is
   * given by the sum of the outputs of the trees whose index is congruent to [i] modulo
   * [num_class].
   *
   * Examples of "grove-per-class" classifier are found in XGBoost, LightGBM, and
   * GradientBoostingClassifier of scikit-learn.
   *
   * The kMultiClfGrovePerClass task type implies the following constraints on the task parameters:
   * output_type=float, grove_per_class=true, num_class>1, leaf_vector_size=1. In addition, we
   * require that the number of trees is evenly divisible by [num_class].
   */
  kMultiClfGrovePerClass = 1,
  /*!
   * \brief The multi-class classification task, in which each tree produces a vector of
   *        probability predictions for all the classes.
   *
   * In this setting, each leaf node in a tree produces a vector output whose length is [num_class].
   * The vector represents probability predictions for all the classes. The outputs of the trees
   * are combined via summing or averaging, depending on the value of the [average_tree_output]
   * field. In effect, each tree is casting a set of weighted (fractional) votes for the classes.
   *
   * An example of kMultiClfProbDistLeaf task type is found in RandomForestClassifier of
   * scikit-learn.
   *
   * The kMultiClfProbDistLeaf task type implies the following constraints on the task parameters:
   * output_type=float, grove_per_class=false, num_class>1, leaf_vector_size=num_class.
   */
  kMultiClfProbDistLeaf = 2,
  /*!
   * \brief The multi-class classification task, in which each tree produces a single integer output
   *        representing an unweighted vote for a particular class.
   *
   * In this setting, each leaf node in a tree produces a single integer output between 0 and
   * [num_class-1] that indicates a vote for a particular class. The outputs of the trees are
   * combined by summing one_hot(tree(i)), where one_hot(x) represents the one-hot-encoded vector
   * with 1 in index [x] and 0 everywhere else, and tree(i) is the output from the i-th tree.
   * Models of type kMultiClfCategLeaf can be converted into the kMultiClfProbDistLeaf type, by
   * converting the output of every leaf node into the equivalent one-hot-encoded vector.
   *
   * An example of kMultiClfCategLeaf task type is found in RandomForestClassifier of cuML.
   *
   * The kMultiClfCategLeaf task type implies the following constraints on the task parameters:
   * output_type=int, grove_per_class=false, num_class>1, leaf_vector_size=1.
   */
  kMultiClfCategLeaf = 3
};

/*! \brief Group of parameters that are dependent on the choice of the task type. */
struct TaskParameter {
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

static_assert(std::is_pod<TaskParameter>::value, "TaskParameter must be POD type");

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
      ThresholdType threshold;   // for non-leaf nodes
    };
    /*! \brief pointer to left and right children */
    int32_t cleft_, cright_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    uint32_t sindex_;
    /*! \brief storage for leaf value or decision threshold */
    Info info_;
    /*!
     * \brief number of data points whose traversal paths include this node.
     *        LightGBM models natively store this statistics.
     */
    uint64_t data_count_;
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
  };

  static_assert(std::is_pod<Node>::value, "Node must be a POD type");
  static_assert(std::is_same<ThresholdType, float>::value
                || std::is_same<ThresholdType, double>::value,
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

  Tree() = default;
  ~Tree() = default;
  Tree(const Tree&) = delete;
  Tree& operator=(const Tree&) = delete;
  Tree(Tree&&) noexcept = default;
  Tree& operator=(Tree&&) noexcept = default;

  inline Tree<ThresholdType, LeafOutputType> Clone() const;

  inline const char* GetFormatStringForNode();
  inline void GetPyBuffer(std::vector<PyBufferFrame>* dest);
  inline void InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                               std::vector<PyBufferFrame>::iterator end,
                               bool assume_ownership);

 private:
  // vector of nodes
  ContiguousArray<Node> nodes_;
  ContiguousArray<LeafOutputType> leaf_vector_;
  ContiguousArray<std::size_t> leaf_vector_offset_;
  ContiguousArray<uint32_t> matching_categories_;
  ContiguousArray<std::size_t> matching_categories_offset_;

  // allocate a new node
  inline int AllocNode();

 public:
  /*! \brief number of nodes */
  int num_nodes;
  /*! \brief initialize the model with a single root node */
  inline void Init();
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid);

  /*!
   * \brief get list of all categorical features that have appeared anywhere in tree
   * \return list of all categorical features used
   */
  inline std::vector<unsigned> GetCategoricalFeatures() const;

  /** Getters **/
  /*!
   * \brief index of the node's left child
   * \param nid ID of node being queried
   */
  inline int LeftChild(int nid) const {
    return nodes_.at(nid).cleft_;
  }
  /*!
   * \brief index of the node's right child
   * \param nid ID of node being queried
   */
  inline int RightChild(int nid) const {
    return nodes_.at(nid).cright_;
  }
  /*!
   * \brief index of the node's "default" child, used when feature is missing
   * \param nid ID of node being queried
   */
  inline int DefaultChild(int nid) const {
    return DefaultLeft(nid) ? LeftChild(nid) : RightChild(nid);
  }
  /*!
   * \brief feature index of the node's split condition
   * \param nid ID of node being queried
   */
  inline uint32_t SplitIndex(int nid) const {
    return (nodes_.at(nid).sindex_ & ((1U << 31U) - 1U));
  }
  /*!
   * \brief whether to use the left child node, when the feature in the split condition is missing
   * \param nid ID of node being queried
   */
  inline bool DefaultLeft(int nid) const {
    return (nodes_.at(nid).sindex_ >> 31U) != 0;
  }
  /*!
   * \brief whether the node is leaf node
   * \param nid ID of node being queried
   */
  inline bool IsLeaf(int nid) const {
    return nodes_.at(nid).cleft_ == -1;
  }
  /*!
   * \brief get leaf value of the leaf node
   * \param nid ID of node being queried
   */
  inline LeafOutputType LeafValue(int nid) const {
    return (nodes_.at(nid).info_).leaf_value;
  }
  /*!
   * \brief get leaf vector of the leaf node; useful for multi-class random forest classifier
   * \param nid ID of node being queried
   */
  inline std::vector<LeafOutputType> LeafVector(int nid) const {
    const std::size_t offset_begin = leaf_vector_offset_.at(nid);
    const std::size_t offset_end = leaf_vector_offset_.at(nid + 1);
    if (offset_begin >= leaf_vector_.Size() || offset_end > leaf_vector_.Size()) {
      // Return empty vector, to indicate the lack of leaf vector
      return std::vector<LeafOutputType>();
    }
    return std::vector<LeafOutputType>(&leaf_vector_[offset_begin],
                                       &leaf_vector_[offset_end]);
      // Use unsafe access here, since we may need to take the address of one past the last
      // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief tests whether the leaf node has a non-empty leaf vector
   * \param nid ID of node being queried
   */
  inline bool HasLeafVector(int nid) const {
    return leaf_vector_offset_.at(nid) != leaf_vector_offset_.at(nid + 1);
  }
  /*!
   * \brief get threshold of the node
   * \param nid ID of node being queried
   */
  inline ThresholdType Threshold(int nid) const {
    return (nodes_.at(nid).info_).threshold;
  }
  /*!
   * \brief get comparison operator
   * \param nid ID of node being queried
   */
  inline Operator ComparisonOp(int nid) const {
    return nodes_.at(nid).cmp_;
  }
  /*!
   * \brief Get list of all categories belonging to the left/right child node. See the
   *        categories_list_right_child_ field of each split to determine whether this list represents
   *        the right child node or the left child node. Categories are integers ranging from 0 to
   *        (n-1), where n is the number of categories in that particular feature. This list is
   *        assumed to be in ascending order.
   * \param nid ID of node being queried
   */
  inline std::vector<uint32_t> MatchingCategories(int nid) const {
    const std::size_t offset_begin = matching_categories_offset_.at(nid);
    const std::size_t offset_end = matching_categories_offset_.at(nid + 1);
    if (offset_begin >= matching_categories_.Size() || offset_end > matching_categories_.Size()) {
      // Return empty vector, to indicate the lack of any matching categories
      // The node might be a numerical split
      return std::vector<uint32_t>();
    }
    return std::vector<uint32_t>(&matching_categories_[offset_begin],
                                 &matching_categories_[offset_end]);
      // Use unsafe access here, since we may need to take the address of one past the last
      // element, to follow with the range semantic of std::vector<>.
  }
  /*!
   * \brief tests whether the node has a non-empty list for matching categories. See
   *        MatchingCategories() for the definition of matching categories.
   * \param nid ID of node being queried
   */
  inline bool HasMatchingCategories(int nid) const {
    return matching_categories_offset_.at(nid) != matching_categories_offset_.at(nid + 1);
  }
  /*!
   * \brief get feature split type
   * \param nid ID of node being queried
   */
  inline SplitFeatureType SplitType(int nid) const {
    return nodes_.at(nid).split_type_;
  }
  /*!
   * \brief test whether this node has data count
   * \param nid ID of node being queried
   */
  inline bool HasDataCount(int nid) const {
    return nodes_.at(nid).data_count_present_;
  }
  /*!
   * \brief get data count
   * \param nid ID of node being queried
   */
  inline uint64_t DataCount(int nid) const {
    return nodes_.at(nid).data_count_;
  }

  /*!
   * \brief test whether this node has hessian sum
   * \param nid ID of node being queried
   */
  inline bool HasSumHess(int nid) const {
    return nodes_.at(nid).sum_hess_present_;
  }
  /*!
   * \brief get hessian sum
   * \param nid ID of node being queried
   */
  inline double SumHess(int nid) const {
    return nodes_.at(nid).sum_hess_;
  }
  /*!
   * \brief test whether this node has gain value
   * \param nid ID of node being queried
   */
  inline bool HasGain(int nid) const {
    return nodes_.at(nid).gain_present_;
  }
  /*!
   * \brief get gain value
   * \param nid ID of node being queried
   */
  inline double Gain(int nid) const {
    return nodes_.at(nid).gain_;
  }
  /*!
   * \brief test whether the list given by MatchingCategories(nid) is associated with the right
   *        child node or the left child node
   * \param nid ID of node being queried
   */
  inline bool CategoriesListRightChild(int nid) const {
    return nodes_.at(nid).categories_list_right_child_;
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
  inline void SetNumericalSplit(int nid, unsigned split_index, ThresholdType threshold,
                                bool default_left, Operator cmp);
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
                                  const std::vector<uint32_t>& categories_list,
                                  bool categories_list_right_child);
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
  inline void SetLeafVector(int nid, const std::vector<LeafOutputType>& leaf_vector);
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

  void ReferenceSerialize(dmlc::Stream* fo) const;
};

struct ModelParam {
  /*!
  * \defgroup model_param
  * Extra parameters for tree ensemble models
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
   * \brief global bias of the model
   *
   * Predicted margin scores of all instances will be adjusted by the global
   * bias. If unspecified, the bias is set to zero.
   */
  float global_bias;
  /*! \} */

  ModelParam() : sigmoid_alpha(1.0f), global_bias(0.0f) {
    std::memset(pred_transform, 0, TREELITE_MAX_PRED_TRANSFORM_LENGTH * sizeof(char));
    std::strncpy(pred_transform, "identity", sizeof(pred_transform));
  }
  ~ModelParam() = default;
  ModelParam(const ModelParam&) = default;
  ModelParam& operator=(const ModelParam&) = default;
  ModelParam(ModelParam&&) = default;
  ModelParam& operator=(ModelParam&&) = default;

  template<typename Container>
  inline std::vector<std::pair<std::string, std::string>>
  InitAllowUnknown(const Container &kwargs);
  inline std::map<std::string, std::string> __DICT__() const;
};

static_assert(std::is_standard_layout<ModelParam>::value,
              "ModelParam must be in the standard layout");

inline void InitParamAndCheck(ModelParam* param,
                              const std::vector<std::pair<std::string, std::string>>& cfg);

/*! \brief thin wrapper for tree ensemble model */
class Model {
 public:
  /*! \brief disable copy; use default move */
  Model() : major_ver_(TREELITE_VER_MAJOR), minor_ver_(TREELITE_VER_MINOR),
    patch_ver_(TREELITE_VER_PATCH) {}
  virtual ~Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
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
  virtual void ReferenceSerialize(dmlc::Stream* fo) const = 0;

  /* In-memory serialization, zero-copy */
  inline std::vector<PyBufferFrame> GetPyBuffer();
  inline static std::unique_ptr<Model> CreateFromPyBuffer(std::vector<PyBufferFrame> frames);

  /* Serialization to a file stream */
  void Serialize(FILE* dest_fp);
  static std::unique_ptr<Model> Deserialize(FILE* src_fp);

  /*!
   * \brief number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  int num_feature;
  /*! \brief Task type */
  TaskType task_type;
  /*! \brief whether to average tree outputs */
  bool average_tree_output;
  /*! \brief Group of parameters that are specific to the particular task type */
  TaskParameter task_param;
  /*! \brief extra parameters */
  ModelParam param;

 private:
  int major_ver_, minor_ver_, patch_ver_;
  TypeInfo threshold_type_;
  TypeInfo leaf_output_type_;
  // Internal functions for serialization
  virtual void GetPyBuffer(std::vector<PyBufferFrame>* dest) = 0;
  virtual void InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                                std::vector<PyBufferFrame>::iterator end,
                                bool assume_ownership) = 0;
  inline static std::unique_ptr<Model> CreateFromPyBufferImpl(
      std::vector<PyBufferFrame> frames, bool assume_ownership);
};

template <typename ThresholdType, typename LeafOutputType>
class ModelImpl : public Model {
 public:
  /*! \brief member trees */
  std::vector<Tree<ThresholdType, LeafOutputType>> trees;

  /*! \brief disable copy; use default move */
  ModelImpl() = default;
  ~ModelImpl() override = default;
  ModelImpl(const ModelImpl&) = delete;
  ModelImpl& operator=(const ModelImpl&) = delete;
  ModelImpl(ModelImpl&&) noexcept = default;
  ModelImpl& operator=(ModelImpl&&) noexcept = default;

  void ReferenceSerialize(dmlc::Stream* fo) const override;
  inline std::size_t GetNumTree() const override {
    return trees.size();
  }
  void SetTreeLimit(std::size_t limit) override {
    return trees.resize(limit);
  }

  inline void GetPyBuffer(std::vector<PyBufferFrame>* dest) override;
  // Set assume_ownership=true to transfer the ownership of the underlying buffers of the frames to
  // the Model object, so that the Model object is responsible for freeing the buffers.
  inline void InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                               std::vector<PyBufferFrame>::iterator end,
                               bool assume_ownership) override;
};

}  // namespace treelite

#include "tree_impl.h"

#endif  // TREELITE_TREE_H_
