package ml.dmlc.treelite4j.java;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Interface to specify a single data entry.
 * @author Philip Cho
 */
public interface Data {
  /**
   * Assign a floating-point value to the entry.
   * @param val value to set
   */
  public void setFValue(float val);
  /**
   * Assign an integer value to the entry. This is useful when feature values
   * and split thresholds are quantized into integers.
   * @param val value to set
   */
  public void setQValue(int val);
  /**
   * Designate the entry as missing.
   */
  public void setMissing();
  /**
   * Test whether the entry is missing.
   * @return whether the entry is missing
   */
  public boolean isMissing();
  /**
   * Obtain the floating-point value stored by the entry.
   * @return floating-point value
   */
  public float getFValue();
  /**
   * Obtain the integer value stored by the entry.
   * @return integer value
   */
  public int getQValue();

  /**
   * Serialize
   * @param out
   * @throws IOException
   */
  public void write(OutputStream out) throws IOException;
}
