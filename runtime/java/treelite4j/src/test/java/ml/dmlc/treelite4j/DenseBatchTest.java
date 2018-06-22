package ml.dmlc.treelite4j;

import org.junit.Test;
import junit.framework.TestCase;
import java.util.concurrent.ThreadLocalRandom;

/**
 * test cases for dense batch
 *
 * @author Philip Cho
 */
public class DenseBatchTest {
  @Test
  public void testDenseBatchBasic() throws TreeliteError {
    for (int i = 0; i < 100000; ++i) {
      int num_row = ThreadLocalRandom.current().nextInt(1, 100);
      int num_col = ThreadLocalRandom.current().nextInt(1, 100);
      float[] data = new float[num_row * num_col];
      for (int k = 0; k < num_row * num_col; ++k) {
        data[k] = ThreadLocalRandom.current().nextFloat() - 0.5f;
      }
      DenseBatch batch = new DenseBatch(data, Float.NaN, num_row, num_col);
      long[] out_num_row = new long[1];
      long[] out_num_col = new long[1];
      TreeliteJNI.checkCall(TreeliteJNI.TreeliteBatchGetDimension(
        batch.getHandle(), false, out_num_row, out_num_col));
      TestCase.assertEquals(num_row, out_num_row[0]);
      TestCase.assertEquals(num_col, out_num_col[0]);
    }
  }
}
