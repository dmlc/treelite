package ml.dmlc.treelite4j;

/**
 * 2D sparse batch, laid out in CSR (Compressed Sparse Row) layout
 * @author Philip Cho
 */
class SparseBatch {
  private float[] data;
  private int[] col_ind;
  private long[] row_ptr;
  private int num_row;
  private int num_col;

  private long handle;

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

  public long getHandle() {
    return this.handle;
  }

  @Override
  protected void finalize() throws Throwable {
    super.finalize();
    dispose();
  }

  public synchronized void dispose() {
    if (handle != 0L) {
      TreeliteJNI.TreeliteDeleteSparseBatch(handle);
      handle = 0;
    }
  }
}
