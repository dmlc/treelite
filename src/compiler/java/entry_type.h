const char* entry_type_template =
R"TREELITETEMPLATE(
package {java_package};

import javolution.io.Union;

public class Entry extends Union {{
  public Signed32 missing = new Signed32();
  public Float32  fvalue  = new Float32();
  public Signed32 qvalue  = new Signed32();
}}
)TREELITETEMPLATE";
