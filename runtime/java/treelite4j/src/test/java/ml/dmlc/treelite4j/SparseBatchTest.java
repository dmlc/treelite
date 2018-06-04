package ml.dmlc.treelite4j;

import org.apache.commons.lang.ArrayUtils;
import org.junit.Test;
import junit.framework.TestCase;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

/**
 * test cases for sparse batch
 *
 * @author Philip Cho
 */
public class SparseBatchTest {
  @Test
  public void testSparseBatchBasic() throws TreeliteError {
    float kDensity = 0.1f;  // % of nonzeros in matrix
    float kProbNextRow = 0.1f;  // transition probability from one row to next
    for (int case_id = 0; case_id < 100000; ++case_id) {
      int num_row = ThreadLocalRandom.current().nextInt(1, 100);
      int num_col = ThreadLocalRandom.current().nextInt(1, 100);
      int nnz = (int)(num_row * num_col * kDensity);
      float[] data = new float[nnz];
      int[] col_ind = new int[nnz];
      ArrayList<Long> row_ptr = new ArrayList<Long>();
      row_ptr.add(0L);
      for (int k = 0; k < data.length; ++k) {
        data[k] = ThreadLocalRandom.current().nextFloat() - 0.5f;
        col_ind[k] = ThreadLocalRandom.current().nextInt(0, num_col);
        if (ThreadLocalRandom.current().nextFloat() < kProbNextRow
            || k == data.length - 1) {
          Arrays.sort(col_ind, row_ptr.get(row_ptr.size()-1).intValue(), k+1);
          row_ptr.add(k + 1L);
        }
      }
      long[] row_ptr_arr
        = ArrayUtils.toPrimitive(row_ptr.toArray(new Long[row_ptr.size()]));
      SparseBatch batch
        = new SparseBatch(data, col_ind, row_ptr_arr, num_row, num_col);
      long[] out_num_row = new long[1];
      long[] out_num_col = new long[1];
      TreeliteJNI.checkCall(TreeliteJNI.TreeliteBatchGetDimension(
        batch.getHandle(), true, out_num_row, out_num_col));
      TestCase.assertEquals(num_row, out_num_row[0]);
      TestCase.assertEquals(num_col, out_num_col[0]);
    }
  }
}
