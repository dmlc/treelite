package ml.dmlc.treelite4j;

/**
 * 2D dense batch, laid out in row-major layout
 * @author Philip Cho
 */
public class DenseBatch {
  private float[] data;
  private float missing_value;
  private int num_row;
  private int num_col;

  private long handle;

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
      TreeliteJNI.TreeliteDeleteDenseBatch(handle, this.data);
      handle = 0;
    }
  }
}
