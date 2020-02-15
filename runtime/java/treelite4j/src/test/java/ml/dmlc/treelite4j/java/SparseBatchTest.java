package ml.dmlc.treelite4j.java;

import junit.framework.TestCase;
import ml.dmlc.treelite4j.DataPoint;
import org.apache.commons.lang.ArrayUtils;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Test cases for sparse batch
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
        = ArrayUtils.toPrimitive(row_ptr.toArray(new Long[0]));
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

  @Test
  public void testSparseBatchBuilder() throws TreeliteError, IOException {
    List<DataPoint> dmat = new ArrayList<DataPoint>() {
      {
        add(new DataPoint(new int[]{0, 1},    new float[]{10f, 20f}));
        add(new DataPoint(new int[]{1, 3},    new float[]{30f, 40f}));
        add(new DataPoint(new int[]{2, 3, 4}, new float[]{50f, 60f, 70f}));
        add(new DataPoint(new int[]{5},       new float[]{80f}));
      }
    };
    SparseBatch batch = BatchBuilder.CreateSparseBatch(dmat);

    // should get 4-by-6 matrix
    long[] out_num_row = new long[1];
    long[] out_num_col = new long[1];
    TreeliteJNI.checkCall(TreeliteJNI.TreeliteBatchGetDimension(
      batch.getHandle(), true, out_num_row, out_num_col));
    TestCase.assertEquals(4, out_num_row[0]);
    TestCase.assertEquals(6, out_num_col[0]);
  }
}
