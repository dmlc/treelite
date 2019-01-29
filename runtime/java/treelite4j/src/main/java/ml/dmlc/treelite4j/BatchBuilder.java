package ml.dmlc.treelite4j;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang.ArrayUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BatchBuilder {
  public static SparseBatch CreateSparseBatch(List<DataPoint> dmat)
      throws TreeliteError, IOException {
    ArrayList<Float> data = new ArrayList<Float>();
    ArrayList<Integer> col_ind = new ArrayList<Integer>();
    ArrayList<Long> row_ptr = new ArrayList<Long>();
    int num_row = dmat.size();
    int num_col = 0;
    row_ptr.add(0L);
    for (DataPoint inst : dmat) {
      int nnz = 0; // count number of nonzero feature values for current row
      for (float e : inst.values()) {
        data.add(e);
        ++nnz;
      }
      if (inst.indices() != null) {
        for (int e : inst.indices()) {
          col_ind.add(e);
          num_col = Math.max(num_col, e + 1);
        }
      } else {
        // when indices are missing, assume feature indices 0, 1, 2, ...
        int num_values = inst.values().length;
        for (int i = 0; i < num_values; ++i) {
          col_ind.add(i);
        }
        num_col = Math.max(num_col, num_values);
      }
      row_ptr.add(row_ptr.get(row_ptr.size() - 1) + (long)nnz);
    }
    float[] data_arr
      = ArrayUtils.toPrimitive(data.toArray(new Float[0]));
    int[] col_ind_arr
      = ArrayUtils.toPrimitive(col_ind.toArray(new Integer[0]));
    long[] row_ptr_arr
      = ArrayUtils.toPrimitive(row_ptr.toArray(new Long[0]));
    return new SparseBatch(data_arr, col_ind_arr, row_ptr_arr, num_row, num_col);
  }

  public static DenseBatch CreateDenseBatch(List<DataPoint> dmat)
      throws TreeliteError, IOException {
    int num_row = dmat.size();
    int num_col = 0;
    // compute num_col
    for (DataPoint inst : dmat) {
      if (inst.indices() != null) {
        for (int e : inst.indices()) {
          num_col = Math.max(num_col, e + 1);
        }
      } else {
        // when indices are missing, assume feature indices 0, 1, 2, ...
        num_col = Math.max(num_col, inst.values().length);
      }
    }
    float[] data = new float[num_row * num_col];
    Arrays.fill(data, Float.NaN);

    /* write nonzero entries */
    int row_id = 0;
    for (DataPoint inst : dmat) {
      if (inst.indices() != null) {
        int[] indices = inst.indices();
        float[] values = inst.values();
        for (int i = 0; i < indices.length; ++i) {
          data[row_id * num_col + indices[i]] = values[i];
        }
      } else {
        // when indices are missing, assume feature indices 0, 1, 2, ...
        float[] values = inst.values();
        int num_values = values.length;
        for (int i = 0; i < num_values; ++i) {
          data[row_id * num_col + i] = values[i];
        }
      }
      ++row_id;
    }
    assert row_id == num_row;

    return new DenseBatch(data, Float.NaN, num_row, num_col);
  }

  public static List<DataPoint> LoadDatasetFromLibSVM(String filename)
      throws TreeliteError, IOException {
    File file = new File(filename);
    LineIterator it = FileUtils.lineIterator(file, "UTF-8");
    ArrayList<DataPoint> dmat = new ArrayList<DataPoint>();
    try {
      while (it.hasNext()) {
        String line = it.nextLine();
        String[] tokens = line.split(" ");
        /* feature indices and values for the data point */
        ArrayList<Integer> indices = new ArrayList<Integer>();
        ArrayList<Float> values = new ArrayList<Float>();
        // ignore label; just read feature values
        for (int i = 1; i < tokens.length; ++i) {
          String[] subtokens = tokens[i].split(":");
          indices.add(Integer.parseInt(subtokens[0]));
          values.add(Float.parseFloat(subtokens[1]));
        }
        dmat.add(new DataPoint(
            ArrayUtils.toPrimitive(indices.toArray(new Integer[0])),
            ArrayUtils.toPrimitive(values.toArray(new Float[0]))));
      }
    } finally {
      it.close();
    }
    return dmat;
  }
}
