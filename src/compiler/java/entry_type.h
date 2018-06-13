const char* entry_type_template =
R"TREELITETEMPLATE(
package {java_package};

import javolution.io.Union;
import ml.dmlc.treelite.Data;

public class Entry extends Union implements Data {{
  public Signed32 missing = new Signed32();
  public Float32  fvalue  = new Float32();
  public Signed32 qvalue  = new Signed32();

  public void setMissing() {{
    this.missing.set(-1);
  }}
  public boolean isMissing() {{
    return this.missing.get() == -1;
  }}
  public void setFValue(float val) {{
    this.fvalue.set(val);
  }}
  public float getFValue() {{
    return this.fvalue.get();
  }}
  public void setQValue(int val) {{
      this.qvalue.set(val);
  }}
  public int getQValue() {{
       return this.qvalue.get();
  }}
}}
)TREELITETEMPLATE";
