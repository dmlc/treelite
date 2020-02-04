package ml.dmlc.treelite4j;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Treelite prediction runtime JNI functions
 * @author Philip Cho
 */
class TreeliteJNI {
  private static final Log logger = LogFactory.getLog(TreeliteJNI.class);
  static {
    try {
      NativeLibLoader.initTreeliteRuntime();
    } catch (Exception ex) {
      logger.error("Failed to load native library", ex);
      throw new RuntimeException(ex);
    }
  }

  /**
   * Check the return code of the JNI call.
   *
   * @throws TreeliteError if the call failed.
   */
  static void checkCall(int ret) throws TreeliteError {
    if (ret != 0) {
      throw new TreeliteError(TreeliteGetLastError());
    }
  }

  public final static native String TreeliteGetLastError();

  public final static native int TreeliteAssembleSparseBatch(
    float[] data, int[] col_ind, long[] row_ptr, long num_row, long num_col,
    long[] out);

  public final static native int TreeliteDeleteSparseBatch(
    long handle, float[] data, int[] col_ind, long[] row_ptr);

  public final static native int TreeliteAssembleDenseBatch(
    float[] data, float missing_value, long num_row, long num_col, long[] out);

  public final static native int TreeliteDeleteDenseBatch(
    long handle, float[] data);

  public final static native int TreeliteBatchGetDimension(
    long handle, boolean batch_sparse, long[] out_num_row, long[] out_num_col);

  public final static native int TreelitePredictorLoad(
    String library_path, int num_worker_thread, long[] out);

  public final static native int TreelitePredictorPredictBatch(
    long handle, long batch, boolean batch_sparse, boolean verbose,
    boolean pred_margin, float[] out_result, long[] out_result_size);

  public final static native int TreelitePredictorPredictInst(
    long handle, byte[] inst, boolean pred_margin, float[] out_result,
    long[] out_result_size);

  public final static native int TreelitePredictorQueryResultSize(
    long handle, long batch, boolean batch_sparse, long[] out);

  public final static native int TreelitePredictorQueryResultSizeSingleInst(
    long handle, long[] out);

  public final static native int TreelitePredictorQueryNumOutputGroup(
    long handle, long[] out);

  public final static native int TreelitePredictorQueryNumFeature(
    long handle, long[] out);

  public final static native int TreelitePredictorQueryPredTransform(
    long handle, String[] out);

  public final static native int TreelitePredictorQuerySigmoidAlpha(
    long handle, float[] out);

  public final static native int TreelitePredictorQueryGlobalBias(
    long handle, float[] out);

  public final static native int TreelitePredictorFree(long handle);

}
