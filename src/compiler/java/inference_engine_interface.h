const char* inference_engine_interface =
R"TREELITETEMPLATE(
package ml.dmlc.treelite4j;

public interface InferenceEngine {
    public int get_num_feature();
    public static float predict(Data[] data, boolean pred_margin);
}
)TREELITETEMPLATE";
