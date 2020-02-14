package ml.dmlc.treelite4j.java;

/**
 * 2D sparse batch, laid out in CSR (Compressed Sparse Row) layout
 * @author Philip Cho
 */
public class SparseBatch {
  private float[] data;
  private int[] col_ind;
  private long[] row_ptr;
  private int num_row;
  private int num_col;

  private long handle;

  /**
   * Create a sparse batch representing a 2D sparse matrix
   * @param data nonzero (non-missing) entries
   * @param col_ind corresponding column indices, should be of same length as
   *                ``data``
   * @param row_ptr offsets to define each instance, should be of length
   *                ``[num_row]+1``
   * @param num_row number of rows (data instances) in the matrix
   * @param num_row number of columns (features) in the matrix
   * @return Created sparse batch
   * @throws TreeliteError
   */
  public SparseBatch(
    float[] data, int[] col_ind, long[] row_ptr, int num_row, int num_col)
      throws TreeliteError {
    this.data = data;
    this.col_ind = col_ind;
    this.row_ptr = row_ptr;
    this.num_row = num_row;
    this.num_col = num_col;

    long[] out = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteAssembleSparseBatch(
      this.data, this.col_ind, this.row_ptr, this.num_row, this.num_col, out));
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
      TreeliteJNI.TreeliteDeleteSparseBatch(
        handle, this.data, this.col_ind, this.row_ptr);
      handle = 0;
    }
  }
}
