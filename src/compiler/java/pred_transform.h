#include <treelite/common.h>
#include <fmt/format.h>

using namespace fmt::literals;

namespace treelite {
namespace compiler {
namespace pred_transform {
namespace java {

inline std::string identity(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(  private static float pred_transform(float margin) {{
    return margin;
  }})TREELITETEMPLATE");
}

inline std::string sigmoid(const Model& model) {
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "sigmoid: alpha must be strictly positive";
  return fmt::format(
R"TREELITETEMPLATE(  private static float pred_transform(float margin) {{
    final double alpha = {alpha};
    return (float)(1.0 / (1.0 + Math.exp(-alpha * margin)));
  }})TREELITETEMPLATE",
    "alpha"_a = alpha);
}

inline std::string exponential(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(  private static float pred_transform(float margin) {{
    return (float)Math.exp(margin);
  }})TREELITETEMPLATE");
}

inline std::string logarithm_one_plus_exp(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(  private static float pred_transform(float margin) {{
    return (float)Math.log1p(Math.exp(margin));
  }})TREELITETEMPLATE");
}

inline std::string identity_multiclass(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "identity_multiclass: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(  private static long pred_transform(float[] pred) {{
    return {num_class};
  }})TREELITETEMPLATE",
      "num_class"_a = model.num_output_group);
}

inline std::string max_index(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "max_index: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(  private static long pred_transform(float[] pred) {{
    final int num_class = {num_class};
    int max_index = 0;
    float max_margin = pred[0];
    for (int k = 1; k < num_class; ++k) {{
      if (pred[k] > max_margin) {{
        max_margin = pred[k];
        max_index = k;
      }}
    }}
    pred[0] = (float)max_index;
    return 1;
  }})TREELITETEMPLATE",
      "num_class"_a = model.num_output_group);
}

inline std::string softmax(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "softmax: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(  private static long pred_transform(float[] pred) {{
    final int num_class = {num_class};
    float max_margin = pred[0];
    double norm_const = 0.0;
    double t;
    for (int k = 1; k < num_class; ++k) {{
      if (pred[k] > max_margin) {{
        max_margin = pred[k];
      }}
    }}
    for (int k = 0; k < num_class; ++k) {{
      t = Math.exp(pred[k] - max_margin);
      norm_const += t;
      pred[k] = (float)t;
    }}
    for (int k = 0; k < num_class; ++k) {{
      pred[k] /= (float)norm_const;
    }}
    return (long)num_class;
  }})TREELITETEMPLATE",
      "num_class"_a = model.num_output_group);
}

inline std::string multiclass_ova(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "multiclass_ova: model is not a proper multi-class classifier";
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "multiclass_ova: alpha must be strictly positive";
  return fmt::format(
R"TREELITETEMPLATE(  private static long pred_transform(float[] pred) {{
    final float alpha = (float){alpha};
    final int num_class = {num_class};
    for (int k = 0; k < num_class; ++k) {{
      pred[k] = (float)(1.0 / (1.0 + Math.exp(-alpha * pred[k])));
    }}
    return (long)num_class;
  }})TREELITETEMPLATE",
      "num_class"_a = model.num_output_group,
      "alpha"_a = alpha);
}

}  // namespace java
}  // namespace pred_transform
}  // namespace compiler
}  // namespace treelite
