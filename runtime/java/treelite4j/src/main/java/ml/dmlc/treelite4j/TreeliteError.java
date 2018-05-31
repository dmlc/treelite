package ml.dmlc.treelite4j;

/**
 * custom error class for Treelite
 *
 * @author Philip Cho
 */
public class TreeliteError extends Exception {
  public TreeliteError(String message) {
    super(message);
  }
}
