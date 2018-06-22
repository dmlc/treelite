const char* data_interface =
R"TREELITETEMPLATE(
package ml.dmlc.treelite4j;

public interface Data {
  public void setMissing();
  public boolean isMissing();
  public void setFValue(float val);
  public float getFValue();
  public void setQValue(int val);
  public int getQValue();
}
)TREELITETEMPLATE";
