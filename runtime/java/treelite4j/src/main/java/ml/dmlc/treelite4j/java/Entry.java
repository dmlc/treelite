package ml.dmlc.treelite4j.java;

import java.nio.ByteOrder;

import javolution.io.Union;

/**
 * A reference implementation for the :java:ref:`Data` interface. This class is used to
 * specify a single data entry. This implementation uses a C-style union so
 * as to save memory space.
 * @author Philip Cho
 */
public class Entry extends Union implements Data {
  /**
   * If ``missing == -1``, the entry is considered missing. Otherwise, the entry
   * is not missing; check the other fields to obtain the actual value of
   * the entry.
   */
  public Signed32 missing = new Signed32();
  /**
   * The value of the entry, in floating-point representation
   */
  public Float32  fvalue  = new Float32();
  /**
   * The value of the entry, in integer representation
   */
  public Signed32 qvalue  = new Signed32();
  /**
   * Assign a floating-point value to the entry.
   * @param val value to set
   */
  public void setFValue(float val) {
    this.fvalue.set(val);
  }
  /**
   * Designate the entry as missing.
   */
  public void setMissing() {
    this.missing.set(-1);
  }
  /**
   * Test whether the entry is missing.
   * @return whether the entry is missing
   */
  public boolean isMissing() {
    return this.missing.get() == -1;
  }
  /**
   * Obtain the floating-point value stored by the entry.
   * @return floating-point value
   */
  public float getFValue() {
    return this.fvalue.get();
  }
  /**
   * The byte-order to use when serializing.
   * @return ``ByteOrder.LITTLE_ENDIAN``, since this class will be
   *         serialized using the litte-endian byte order.
   */
  public ByteOrder byteOrder() {
    return ByteOrder.LITTLE_ENDIAN;  // use little endian when serializing
  }
  /**
   * Assign an integer value to the entry. This is useful when feature values
   * and split thresholds are quantized into integers.
   * @param val value to set
   */
  public void setQValue(int val) {
    this.qvalue.set(val);
  }
  /**
   * Obtain the integer value stored by the entry.
   * @return integer value
   */
  public int getQValue() {
    return this.qvalue.get();
  }
}
