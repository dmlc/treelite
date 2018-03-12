const char* entry_type =
"package treelite.predictor;\n"
"\n"
"import javolution.io.Union;\n"
"\n"
"public class Entry extends Union {\n"
"  Signed32 missing = new Signed32();\n"
"  Float32  fvalue  = new Float32();\n"
"  Signed32 qvalue  = new Signed32();\n"
"}\n";