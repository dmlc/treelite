package ml.dmlc.treelite4j.java;

/**
 * Interface for making inference (WORK-IN-PROGRESS)
 * @author James Liu
 */
public interface InferenceEngine {
  /**
   * Get the number of output groups for the compiled model. This number is
   * 1 for tasks other than multi-class classification. For multi-class
   * classification task, the number is equal to the number of classes.
   * @return Number of output groups
   */
  public int getNumOutputGroup();

  /**
   * Get the number of features used by the compiled model. Call this method
   * to allocate array for storing data entries of a single instance.
   * @return Number of features
   */
  public int getNumFeature();

  /**
   * Perform single-instance prediction
   * @param inst array of data entires(features) comprising the instance
   * @param pred_margin whether to predict a probability or a raw margin score
   * @return Resulting predictions, of dimension ``[num_output_group]``
   */
  public float[] predict(Data[] inst, boolean pred_margin);
}
