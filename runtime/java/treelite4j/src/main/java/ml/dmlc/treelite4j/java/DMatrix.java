package ml.dmlc.treelite4j.java;

import org.apache.commons.lang.ArrayUtils;

import java.util.List;

/**
 * An opaque data matrix class. The actual object is stored in the C++ object handle.
 * @author Hyunsu Cho
 */
public class DMatrix {
  private long num_row, num_col, num_elem;  // dimensions of the data matrix
  private long handle;  // handle to C++ DMatrix object

  /**
   * Create a data matrix representing a 2D sparse matrix
   * @param data nonzero (non-missing) entries, float32 type
   * @param col_ind corresponding column indices, should be of same length as ``data``
   * @param row_ptr offsets to define each instance, should be of length ``[num_row]+1``
   * @param num_row number of rows (data points) in the matrix
   * @param num_col number of columns (features) in the matrix
   * @throws TreeliteError error during matrix construction
   */
  public DMatrix(float[] data, int[] col_ind, long[] row_ptr, long num_row, long num_col)
      throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteDMatrixCreateFromCSRWithFloat32In(
        data, col_ind, row_ptr, num_row, num_col, out));
    this.handle = out[0];
    setDims();
  }

  public DMatrix(double[] data, int[] col_ind, long[] row_ptr, long num_row, long num_col)
      throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteDMatrixCreateFromCSRWithFloat64In(
        data, col_ind, row_ptr, num_row, num_col, out));
    this.handle = out[0];
    setDims();
  }

  /**
   * Create a data matrix representing a 2D dense matrix
   * @param data array of entries, should be of length ``[num_row]*[num_col]`` and of float32 type
   * @param missing_value floating-point value representing a missing value;
   *                      usually set of ``Float.NaN``.
   * @param num_row number of rows (data instances) in the matrix
   * @param num_col number of columns (features) in the matrix
   * @throws TreeliteError error during matrix construction
   */
  public DMatrix(float[] data, float missing_value, long num_row, long num_col)
      throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteDMatrixCreateFromMatWithFloat32In(
        data, num_row, num_col, missing_value, out));
    this.handle = out[0];
    setDims();
  }

  /**
   * Create a data matrix representing a 2D dense matrix (float64 type)
   * @param data array of entries, should be of length ``[num_row]*[num_col]`` and of float64 type
   * @param missing_value floating-point value representing a missing value;
   *                      usually set of ``Double.NaN``.
   * @param num_row number of rows (data instances) in the matrix
   * @param num_col number of columns (features) in the matrix
   * @throws TreeliteError error during matrix construction
   */
  public DMatrix(double[] data, double missing_value, long num_row, long num_col)
      throws TreeliteError {
    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteDMatrixCreateFromMatWithFloat64In(
        data, num_row, num_col, missing_value, out));
    this.handle = out[0];
    setDims();
  }

  private void setDims() throws TreeliteError {
    long[] out_num_row = new long[1];
    long[] out_num_col = new long[1];
    long[] out_num_elem = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteDMatrixGetDimension(
        this.handle, out_num_row, out_num_col, out_num_elem));
    this.num_row = out_num_row[0];
    this.num_col = out_num_col[0];
    this.num_elem = out_num_elem[0];
  }

  /**
   * Get the underlying native handle
   * @return Integer representing memory address
   */
  public long getHandle() {
    return this.handle;
  }

  /**
   * Get the number of rows in the matrix
   * @return Number of rows in the matrix
   */
  public long getNumRow() {
    return this.num_row;
  }

  /**
   * Get the number of columns in the matrix
   * @return Number of columns in the matrix
   */
  public long getNumCol() {
    return this.num_col;
  }

  /**
   * Get the number of elements in the matrix
   * @return Number of elements in the matrix
   */
  public long getNumElements() {
    return this.num_elem;
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();
    dispose();
  }

  /**
   * Destructor, to be called when the object is garbage collected
   */
  public synchronized void dispose() {
    if (this.handle != 0L) {
      TreeliteJNI.TreeliteDMatrixFree(this.handle);
      this.handle = 0;
    }
  }
}
