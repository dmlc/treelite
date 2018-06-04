package ml.dmlc.treelite4j;

import java.nio.ByteOrder;

import javolution.io.Union;

public class Entry extends Union implements Data {
  public Signed32 missing = new Signed32();
  public Float32  fvalue  = new Float32();

  public void setFValue(float val) {
    this.fvalue.set(val);
  }
  public void setMissing() {
    this.missing.set(-1);
  }
  public boolean isMissing() {
    return this.missing.get() == -1;
  }
  public float getFValue() {
    return this.fvalue.get();
  }
  public ByteOrder byteOrder() {  // use little endian when serializing
      return ByteOrder.LITTLE_ENDIAN;
  }
}
