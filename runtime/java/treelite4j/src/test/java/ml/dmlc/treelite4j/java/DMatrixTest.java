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
 * Test cases for data matrix
 *
 * @author Hyunsu Cho
 */
public class DMatrixTest {
  @Test
  public void testDenseDMatrixBasicFloat32() throws TreeliteError {
    for (int i = 0; i < 1000; ++i) {
      int num_row = ThreadLocalRandom.current().nextInt(1, 100);
      int num_col = ThreadLocalRandom.current().nextInt(1, 100);
      float[] data = new float[num_row * num_col];
      for (int k = 0; k < num_row * num_col; ++k) {
        data[k] = ThreadLocalRandom.current().nextFloat() - 0.5f;
      }
      DMatrix dmat = new DMatrix(data, Float.NaN, num_row, num_col);
      TestCase.assertEquals(num_row, dmat.getNumRow());
      TestCase.assertEquals(num_col, dmat.getNumCol());
      TestCase.assertEquals(num_row * num_col, dmat.getNumElements());
    }
  }

  @Test
  public void testSparseDMatrixBasicFloat32() throws TreeliteError {
    float kDensity = 0.1f;  // % of nonzeros in matrix
    float kProbNextRow = 0.1f;  // transition probability from one row to next
    for (int case_id = 0; case_id < 1000; ++case_id) {
      int num_row = ThreadLocalRandom.current().nextInt(1, 100);
      int num_col = ThreadLocalRandom.current().nextInt(1, 100);
      int nnz = (int)(num_row * num_col * kDensity);
      float[] data = new float[nnz];
      int[] col_ind = new int[nnz];
      ArrayList<Long> row_ptr = new ArrayList<Long>();
      row_ptr.add(0L);
      for (int k = 0; k < nnz; ++k) {
        data[k] = ThreadLocalRandom.current().nextFloat() - 0.5f;
        col_ind[k] = ThreadLocalRandom.current().nextInt(0, num_col);
        if (ThreadLocalRandom.current().nextFloat() < kProbNextRow
            && row_ptr.size() < num_row) {
          Arrays.sort(col_ind, row_ptr.get(row_ptr.size() - 1).intValue(), k + 1);
          row_ptr.add(k + 1L);
        }
      }
      Arrays.sort(col_ind, row_ptr.get(row_ptr.size() - 1).intValue(), nnz);
      row_ptr.add((long)nnz);
      while (row_ptr.size() < num_row + 1) {
        row_ptr.add(row_ptr.get(row_ptr.size() - 1));
      }
      TestCase.assertEquals(row_ptr.size(), num_row + 1);
      long[] row_ptr_arr = ArrayUtils.toPrimitive(row_ptr.toArray(new Long[0]));
      DMatrix dmat = new DMatrix(data, col_ind, row_ptr_arr, num_row, num_col);
      TestCase.assertEquals(num_row, dmat.getNumRow());
      TestCase.assertEquals(num_col, dmat.getNumCol());
      TestCase.assertEquals(nnz, dmat.getNumElements());
    }
  }

  @Test
  public void testSparseDMatrixBuilder() throws TreeliteError, IOException {
    List<DataPoint> data_list = new ArrayList<DataPoint>() {
      {
        add(new DataPoint(new int[]{0, 1},    new float[]{10f, 20f}));
        add(new DataPoint(new int[]{1, 3},    new float[]{30f, 40f}));
        add(new DataPoint(new int[]{2, 3, 4}, new float[]{50f, 60f, 70f}));
        add(new DataPoint(new int[]{5},       new float[]{80f}));
      }
    };
    DMatrix dmat = DMatrixBuilder.createSparseCSRDMatrixFloat32(data_list.iterator());

    // should get 4-by-6 matrix
    TestCase.assertEquals(4, dmat.getNumRow());
    TestCase.assertEquals(6, dmat.getNumCol());
  }
}
