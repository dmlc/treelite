#include <treelite/c_api.h>
#include <treelite/c_api_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define safe_treelite(call) {  \
  int err = (call);  \
  if (err == -1) {  \
    fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call,  \
            TreeliteGetLastError());  \
    exit(1);  \
  }  \
}

ModelHandle BuildModel() {
  ModelHandle model;
  ModelBuilderHandle builder;
  TreeBuilderHandle tree;
  float threshold_value = 0.0f;
  float leaf0_value = -1.0f;
  float leaf1_value = 1.0f;
  ValueHandle threshold, leaf0, leaf1;
  safe_treelite(TreeliteTreeBuilderCreateValue(&threshold_value, "float32", &threshold));
  safe_treelite(TreeliteTreeBuilderCreateValue(&leaf0_value, "float32", &leaf0));
  safe_treelite(TreeliteTreeBuilderCreateValue(&leaf1_value, "float32", &leaf1));

  safe_treelite(TreeliteCreateTreeBuilder("float32", "float32", &tree));
  safe_treelite(TreeliteTreeBuilderCreateNode(tree, 0));
  safe_treelite(TreeliteTreeBuilderCreateNode(tree, 1));
  safe_treelite(TreeliteTreeBuilderCreateNode(tree, 2));
  safe_treelite(TreeliteTreeBuilderSetNumericalTestNode(tree, 0, 0, "<", threshold, 1, 1, 2));
  safe_treelite(TreeliteTreeBuilderSetLeafNode(tree, 1, leaf0));
  safe_treelite(TreeliteTreeBuilderSetLeafNode(tree, 2, leaf1));
  safe_treelite(TreeliteTreeBuilderSetRootNode(tree, 0));

  safe_treelite(TreeliteCreateModelBuilder(2, 1, 0, "float32", "float32", &builder));
  safe_treelite(TreeliteModelBuilderInsertTree(builder, tree, -1));
  safe_treelite(TreeliteModelBuilderCommitModel(builder, &model));

  // Clean up
  safe_treelite(TreeliteTreeBuilderDeleteValue(threshold));
  safe_treelite(TreeliteTreeBuilderDeleteValue(leaf0));
  safe_treelite(TreeliteTreeBuilderDeleteValue(leaf1));
  safe_treelite(TreeliteDeleteTreeBuilder(tree));
  safe_treelite(TreeliteDeleteModelBuilder(builder));

  return model;
}

int main() {
  ModelHandle model = BuildModel();
  size_t num_row = 5;
  size_t num_col = 2;
  size_t output_alloc_size;
  float input[10] = {-2.0f, 0.0f,
                     -1.0f, 0.0f,
                      0.0f, 0.0f,
                      1.0f, 0.0f,
                      2.0f, 0.0f};
  float* output;
  size_t out_result_size;
  safe_treelite(TreeliteGTILGetPredictOutputSize(model, num_row, &output_alloc_size));
  output = (float*)malloc(output_alloc_size * sizeof(float));
  safe_treelite(TreeliteGTILPredict(model, input, num_row, output, 0, &out_result_size));

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
