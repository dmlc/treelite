package ml.dmlc.treelite4j;

/**
 * Custom error class for Treelite
 *
 * @author Philip Cho
 */
public class TreeliteError extends Exception {
  private static final long serialVersionUID = 1L;
  public TreeliteError(String message) {
    super(message);
  }
}
