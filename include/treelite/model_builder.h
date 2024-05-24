/*!
 * Copyright (c) 2023 by Contributors
 * \file model_builder.h
 * \brief C++ API for constructing Model objects
 * \author Hyunsu Cho
 */

#ifndef TREELITE_MODEL_BUILDER_H_
#define TREELITE_MODEL_BUILDER_H_

#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace treelite {

class Model;

namespace model_builder {

class Metadata;
class TreeAnnotation;
class PostProcessorFunc;

/*!
 * \brief Model builder interface.
 * \note A model builder object must be accessed by a single thread. For parallel tree construction,
 *       build multiple model objects and then concatenate them.
 */
class ModelBuilder {
 public:
  /*!
   * \brief Set a flag to control validation behavior.
   * Currently, we support "check_orphaned_nodes" (defaults to true).
   * \param flag Name of the flag
   * \param value Value to set the flag
   */
  virtual void SetValidationFlag(std::string const& flag, bool value) = 0;
  /*!
   * \brief Start a new tree
   */
  virtual void StartTree() = 0;
  /*!
   * \brief End the current tree
   */
  virtual void EndTree() = 0;

  /*!
   * \brief Start a new node
   * \param node_key Integer key that unique identifies the node.
   */
  virtual void StartNode(int node_key) = 0;
  /*!
   * \brief End the current node
   */
  virtual void EndNode() = 0;

  /*!
   * \brief Declare the current node as a numerical test node, where the test is of form
   *        [feature value] [cmp] [threshold]. Data points for which the test evaluates to True
   *        will be mapped to the left child node; all other data points (for which the test
   *        evaluates to False) will be mapped to the right child node.
   * \param split_index Feature ID
   * \param threshold Threshold
   * \param default_left Whether the missing value should be mapped to the left child
   * \param cmp Comparison operator
   * \param left_child_key Integer key that unique identifies the left child node.
   * \param right_child_key  Integer key that unique identifies the right child node.
   */
  virtual void NumericalTest(std::int32_t split_index, double threshold, bool default_left,
      Operator cmp, int left_child_key, int right_child_key)
      = 0;
  /*!
   * \brief Declare the current node as a categorical test node, where the test is of form
   *        [feature value] \\in [category list].
   * \param split_index Feature ID
   * \param default_left Whether the missing value should be mapped to the left child
   * \param category_list List of categories to be tested for match
   * \param category_list_right_child Whether the data points for which the test evaluates to True
   *                                  should be mapped to the right child or the left child.
   * \param left_child_key Integer key that unique identifies the left child node.
   * \param right_child_key Integer key that unique identifies the right child node.
   */
  virtual void CategoricalTest(std::int32_t split_index, bool default_left,
      std::vector<std::uint32_t> const& category_list, bool category_list_right_child,
      int left_child_key, int right_child_key)
      = 0;

  /*!
   * \brief Declare the current node as a leaf node with a scalar output
   * \param leaf_value Value of leaf output
   */
  virtual void LeafScalar(double leaf_value) = 0;
  /*!
   * \brief Declare the current node as a leaf node with a vector output
   * \param leaf_vector Value of leaf output
   */
  virtual void LeafVector(std::vector<float> const& leaf_vector) = 0;
  /*!
   * \brief Declare the current node as a leaf node with a vector output
   * \param leaf_vector Value of leaf output
   */
  virtual void LeafVector(std::vector<double> const& leaf_vector) = 0;

  /*!
   * \brief Specify the gain (loss reduction) that's resulted from the current split.
   * \param gain Gain (loss reduction)
   */
  virtual void Gain(double gain) = 0;
  /*!
   * \brief Specify the number of data points (samples) that are mapped to the current node.
   * \param data_count Number of data points
   */
  virtual void DataCount(std::uint64_t data_count) = 0;
  /*!
   * \brief Specify the weighted sample count or the sum of Hessians for the data points that
   *        are mapped to the current node.
   * \param sum_hess Weighted sample count or the sum of Hessians
   */
  virtual void SumHess(double sum_hess) = 0;

  /*!
   * \brief Specify a metadata for this model, if no metadata was previously specified.
   * \param metadata Model metadata
   * \param tree_annotation Annotation for individual trees
   * \param postprocessor Postprocessor for prediction outputs
   * \param base_scores Baseline scores for targets and classes, before adding tree outputs.
   *                    Also known as the intercept.
   * \param attributes Arbitrary JSON object, to be stored in the "attributes" field in the
   *                   model object.
   */
  virtual void InitializeMetadata(Metadata const& metadata, TreeAnnotation const& tree_annotation,
      PostProcessorFunc const& postprocessor, std::vector<double> const& base_scores,
      std::optional<std::string> const& attributes)
      = 0;
  /*!
   * \brief Conclude model building and obtain the final model object.
   * \return Final model object
   */
  virtual std::unique_ptr<Model> CommitModel() = 0;

  virtual ~ModelBuilder() = default;
};

/*!
 * \brief Annotation for individual trees. Use this object to look up which target and class
 *        each tree is associated with.
 *
 * The output of each target / class is obtained by summing the outputs of all trees that are
 * associated with that target / class.
 * target_id[i] indicates the target the i-th tree is associated with.
 * (-1 indicates that the tree is a multi-target tree, whose output gets counted for all targets.)
 * class_id[i] indicates the class the i-th tree is associated with.
 * (-1 indicates that the tree's output gets counted for all classes.)
 */
struct TreeAnnotation {
  std::int32_t num_tree{0};
  std::vector<std::int32_t> target_id{};
  std::vector<std::int32_t> class_id{};
  /*!
   * \brief Constructor for TreeAnnotation object
   * \param num_tree Number of trees
   * \param target_id Target that each tree is associated with (see explanation above)
   * \param class_id Class that each tree is associated with (see explanation above)
   */
  TreeAnnotation(std::int32_t num_tree, std::vector<std::int32_t> const& target_id,
      std::vector<std::int32_t> const& class_id);
};

/*!
 * \brief Parameter type used to configure postprocessor functions.
 */
using PostProcessorConfigParam = std::variant<std::int64_t, double, std::string>;

/*!
 * \brief Specification for postprocessor of prediction outputs.
 */
struct PostProcessorFunc {
  std::string name{};
  std::map<std::string, PostProcessorConfigParam> config{};
  /*!
   * \brief Constructor for PostProcessorFunc object, with no configuration parameters
   * \param name Name of the postprocessor
   */
  explicit PostProcessorFunc(std::string const& name);
  /*!
   * \brief Constructor for PostProcessorFunc object
   * \param name Name of the postprocessor
   * \param config Optional parameters to configure the postprocessor.
   *               Pass an empty map to indicate the lack of parameters.
   */
  PostProcessorFunc(
      std::string const& name, std::map<std::string, PostProcessorConfigParam> const& config);
};

/*!
 * \brief Metadata object, consisting of metadata information about the model at large.
 */
struct Metadata {
  std::int32_t num_feature{0};
  TaskType task_type{TaskType::kRegressor};
  bool average_tree_output{false};
  std::int32_t num_target{1};
  std::vector<std::int32_t> num_class{1};
  std::array<std::int32_t, 2> leaf_vector_shape{1, 1};
  /*!
   * \brief Constructor for Metadata object
   * \param num_feature Number of features
   * \param task_type Task type
   * \param average_tree_output Whether to average outputs of trees
   * \param num_target Number of targets
   * \param num_class Number of classes. num_class[i] is the number of classes of target i.
   * \param leaf_vector_shape Shape of the output from each leaf node
   */
  Metadata(std::int32_t num_feature, TaskType task_type, bool average_tree_output,
      std::int32_t num_target, std::vector<std::int32_t> const& num_class,
      std::array<std::int32_t, 2> const& leaf_vector_shape);
};

/*!
 * \brief Initialize a model builder object with a given set of metadata.
 * \param threshold_type Type of thresholds in the tree model
 * \param leaf_output_type Type of leaf outputs in the tree model
 * \param metadata Model metadata
 * \param tree_annotation Annotation for individual trees
 * \param postprocessor Postprocessor for prediction outputs
 * \param base_scores Baseline scores for targets and classes, before adding tree outputs.
 *                    Also known as the intercept.
 * \param attributes Arbitrary JSON object, to be stored in the "attributes" field in the
 *                   model object.
 * \return Model builder object
 */
std::unique_ptr<ModelBuilder> GetModelBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type,
    Metadata const& metadata, TreeAnnotation const& tree_annotation,
    PostProcessorFunc const& postprocessor, std::vector<double> const& base_scores,
    std::optional<std::string> const& attributes = std::nullopt);

/*!
 * \brief Initialize a model builder object with empty metadata. Remember to provide
 *        metadata later, with a call to InitializeMetadata().
 * \param threshold_type Type of thresholds in the tree model
 * \param leaf_output_type Type of leaf outputs in the tree model
 * \return Model builder object
 */
std::unique_ptr<ModelBuilder> GetModelBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type);

/*!
 * \brief Initialize a model builder object from a JSON string. The JSON string must contain
 *        all relevant metadata. See \ref GetModelBuilder for the list of necessary metadata.
 * \param json_str JSON string containing relevant metadata.
 * \return Model builder object
 */
std::unique_ptr<ModelBuilder> GetModelBuilder(std::string const& json_str);
// Initialize metadata from a JSON string

}  // namespace model_builder
}  // namespace treelite

#endif  // TREELITE_MODEL_BUILDER_H_
