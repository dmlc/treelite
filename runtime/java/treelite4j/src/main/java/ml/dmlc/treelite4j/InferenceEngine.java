package ml.dmlc.treelite4j;

/**
 * Interface for making inference (WORK-IN-PROGRESS)
 * @author James Liu
 */
public interface InferenceEngine {
  /**
   * Get number of features used
   * @return number of features used
   */
  public int get_num_feature();
  /**
   * Make prediction for a single instance
   * @param data single instance consisting of data entries. Should be of
   *             length ``[num_feature]``.
   * @param pred_margin whether to predict probabilities or raw margin scores
   * @return Prediction result
   */
  public float predict(Data[] data, boolean pred_margin);
}
