const char* data_interface =
R"TREELITETEMPLATE(
package ml.dmlc.treelite;

public interface Data {
  public void setFValue(float val);
  public void setMissing();
  public boolean isMissing();
  public float getFValue();
}
)TREELITETEMPLATE";
