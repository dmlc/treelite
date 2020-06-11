package ml.dmlc.treelite4j.java;

/**
 * 2D dense batch, laid out in row-major layout
 * @author Hyunsu Cho
 */
public class DenseBatch {
  private float[] data;
  private float missing_value;
  private int num_row;
  private int num_col;

  private long handle;

  /**
   * Create a dense batch representing a 2D dense matrix
   * @param data array of entries, should be of length ``[num_row]*[num_col]``
   * @param missing_value floating-point value representing a missing value;
   *                      usually set of ``Float.NaN``.
   * @param num_row number of rows (data instances) in the matrix
   * @param num_row number of columns (features) in the matrix
   * @return Created dense batch
   * @throws TreeliteError
   */
  public DenseBatch(
    float[] data, float missing_value, int num_row, int num_col)
      throws TreeliteError {
    this.data = data;
    this.missing_value = missing_value;
    this.num_row = num_row;
    this.num_col = num_col;

    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteAssembleDenseBatch(
      this.data, this.missing_value, this.num_row, this.num_col, out));
    handle = out[0];
  }

  /**
   * Get the underlying native handle
   * @return Integer representing memory address
   */
  public long getHandle() {
    return this.handle;
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
    if (handle != 0L) {
      TreeliteJNI.TreeliteDeleteDenseBatch(handle, this.data);
      handle = 0;
    }
  }
}
