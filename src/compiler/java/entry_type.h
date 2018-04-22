std::string entry_type =
"package " +
param.java_package +
";\n" +
"\n" +
"import javolution.io.Union;\n" +
"\n" +
"public class Entry extends Union {\n" +
"  public Signed32 missing = new Signed32();\n" +
"  public Float32  fvalue  = new Float32();\n" +
"  public Signed32 qvalue  = new Signed32();\n" +
"}\n";