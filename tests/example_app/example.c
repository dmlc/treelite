/*!
 * Copyright (c) 2023 by Contributors
 * \file example.c
 * \brief Test using Treelite as a C++ library
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <treelite/c_api.h>

#define safe_treelite(call)                                                                       \
  {                                                                                               \
    int err = (call);                                                                             \
    if (err == -1) {                                                                              \
      fprintf(                                                                                    \
          stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, TreeliteGetLastError()); \
      exit(1);                                                                                    \
    }                                                                                             \
  }

TreeliteModelHandle BuildModel() {
  TreeliteModelHandle model;
  TreeliteModelBuilderHandle builder;
  char const* model_metadata
      = "{"
        "  \"threshold_type\": \"float32\","
        "  \"leaf_output_type\": \"float32\","
        "  \"metadata\": {"
        "    \"num_feature\": 2,"
        "    \"task_type\": \"kRegressor\","
        "    \"average_tree_output\": false,"
        "    \"num_target\": 1,"
        "    \"num_class\": [1],"
        "    \"leaf_vector_shape\": [1, 1]"
        "  },"
        "  \"tree_annotation\": {"
        "    \"num_tree\": 1,"
        "    \"target_id\": [0],"
        "    \"class_id\": [0]"
        "  },"
        "  \"postprocessor\": {"
        "    \"name\": \"identity\""
        "  },"
        "  \"base_scores\": [0.0]"
        "}";
  safe_treelite(TreeliteGetModelBuilder(model_metadata, &builder));
  safe_treelite(TreeliteModelBuilderStartTree(builder));
  safe_treelite(TreeliteModelBuilderStartNode(builder, 0));
  safe_treelite(TreeliteModelBuilderNumericalTest(builder, 0, 0.0, 0, "<", 1, 2));
  safe_treelite(TreeliteModelBuilderEndNode(builder));
  safe_treelite(TreeliteModelBuilderStartNode(builder, 1));
  safe_treelite(TreeliteModelBuilderLeafScalar(builder, -1.0));
  safe_treelite(TreeliteModelBuilderEndNode(builder));
  safe_treelite(TreeliteModelBuilderStartNode(builder, 2));
  safe_treelite(TreeliteModelBuilderLeafScalar(builder, 1.0));
  safe_treelite(TreeliteModelBuilderEndNode(builder));
  safe_treelite(TreeliteModelBuilderEndTree(builder));

  safe_treelite(TreeliteModelBuilderCommitModel(builder, &model));

  // Clean up
  safe_treelite(TreeliteDeleteModelBuilder(builder));

  return model;
}

int main() {
  TreeliteModelHandle model = BuildModel();
  size_t num_row = 5;
  size_t num_col = 2;
  float input[10] = {-2.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f};
  float* output;
  size_t out_result_size;

  char const* gtil_config_str
      = "{"
        "  \"predict_type\": \"default\","
        "  \"nthread\": 2"
        "}";
  TreeliteGTILConfigHandle gtil_config;
  safe_treelite(TreeliteGTILParseConfig(gtil_config_str, &gtil_config));

  uint64_t const* output_shape;
  uint64_t output_ndim;
  safe_treelite(
      TreeliteGTILGetOutputShape(model, num_row, gtil_config, &output_shape, &output_ndim));
  output = (float*)malloc(output_shape[0] * output_shape[1] * sizeof(float));
  safe_treelite(TreeliteGTILPredict(model, input, "float32", num_row, output, gtil_config));

  printf("TREELITE_VERSION = %s\n", TREELITE_VERSION);

  for (size_t i = 0; i < num_row; ++i) {
    printf("Input %d: [%f", (int)i, input[i * num_col]);
    for (size_t j = 1; j < num_col; ++j) {
      printf(", %f", input[i * num_col + j]);
    }
    printf("], output: %f\n", output[i]);
  }

  free(output);

  return 0;
}
