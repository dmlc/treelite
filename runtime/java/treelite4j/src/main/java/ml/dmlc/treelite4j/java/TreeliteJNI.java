package ml.dmlc.treelite4j.java;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Treelite prediction runtime JNI functions
 * @author Hyunsu Cho
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

  public static native String TreeliteGetLastError();

  public static native int TreeliteDMatrixCreateFromCSRWithFloat32In(
      float[] data, int[] col_ind, long[] row_ptr, long num_row, long num_col, long[] out);

  public static native int TreeliteDMatrixCreateFromCSRWithFloat64In(
      double[] data, int[] col_ind, long[] row_ptr, long num_row, long num_col, long[] out);

  public static native int TreeliteDMatrixCreateFromMatWithFloat32In(
      float[] data, long num_row, long num_col, float missing_value, long[] out);

  public static native int TreeliteDMatrixCreateFromMatWithFloat64In(
      double[] data, long num_row, long num_col, double missing_value, long[] out);

  public static native int TreeliteDMatrixGetDimension(
      long handle, long[] out_num_row, long[] out_num_col, long[] out_nelem);

  public static native int TreeliteDMatrixFree(
      long handle);

  public static native int TreelitePredictorLoad(
      String library_path, int num_worker_thread, long[] out);

  public static native int TreelitePredictorPredictBatchWithFloat32Out(
      long handle, long batch, boolean verbose, boolean pred_margin, float[] out_result,
      long[] out_result_size);

  public static native int TreelitePredictorPredictBatchWithFloat64Out(
      long handle, long batch, boolean verbose, boolean pred_margin, double[] out_result,
      long[] out_result_size);

  public static native int TreelitePredictorPredictBatchWithUInt32Out(
      long handle, long batch, boolean verbose, boolean pred_margin, int[] out_result,
      long[] out_result_size);

  public static native int TreelitePredictorQueryResultSize(
      long handle, long batch, long[] out);

  public static native int TreelitePredictorQueryNumClass(
      long handle, long[] out);

  public static native int TreelitePredictorQueryNumFeature(
      long handle, long[] out);

  public static native int TreelitePredictorQueryPredTransform(
      long handle, String[] out);

  public static native int TreelitePredictorQuerySigmoidAlpha(
      long handle, float[] out);

  public static native int TreelitePredictorQueryRatioC(
      long handle, float[] out);

  public static native int TreelitePredictorQueryGlobalBias(
      long handle, float[] out);

  public static native int TreelitePredictorQueryThresholdType(
      long handle, String[] out);

  public static native int TreelitePredictorQueryLeafOutputType(
      long handle, String[] out);

  public static native int TreelitePredictorFree(long handle);

}
