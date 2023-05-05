"""Warning for deprecated C codegen functionalities"""
import warnings

def deprecate(subject):
    """Generate a warning with a subject"""
    warnings.warn(
        f"{subject} is deprecated and scheduled for removal in Treelite 4.0. "
        "Please use TL2cgen instead. "
        "Consult the migration guide at "
        "https://tl2cgen.readthedocs.io/en/latest/treelite-migration.html."
    )
