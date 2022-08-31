import textwrap
import treelite
from compatibility_tester import _train_model, _fetch_data
from serializer import treelite_serialize


def inspect_frames():
    X, y = _fetch_data()
    clf = _train_model(X, y)
    tl_model = treelite.sklearn.import_model(clf)
    frames = treelite_serialize(tl_model)
    for idx, (format_str, itemsize, frame) in enumerate(
        zip(
            frames["header"]["format_str"],
            frames["header"]["itemsize"],
            frames["frames"],
        )
    ):
        print(f"Frame {idx}: {format_str} {itemsize}")
        print(textwrap.indent(str(frame), "    "))


if __name__ == "__main__":
    inspect_frames()
