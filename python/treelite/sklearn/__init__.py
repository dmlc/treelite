"""Model loader ingest scikit-learn models into Treelite"""

from .importer import import_model


def import_model_with_model_builder(sklearn_model):
    """Stub for removed function"""
    raise NotImplementedError(
        "treelite.sklearn.import_model_with_model_builder() was removed in Treelite 4.0. "
        "Please use treelite.sklearn.import_model() instead."
    )


__all__ = ["import_model", "import_model_with_model_builder"]
