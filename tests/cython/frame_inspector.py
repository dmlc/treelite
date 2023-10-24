import textwrap

from compatibility_tester import _fetch_data, _train_model
from serializer import treelite_serialize

import treelite


def inspect_frames():
    X, y = _fetch_data()
    clf = _train_model(X, y)
    tl_model = treelite.frontend.from_lightgbm(clf.booster_)
    frames = treelite_serialize(tl_model)
    for idx, (format_str, itemsize, frame) in enumerate(
        zip(
            frames["header"]["format_str"],
            frames["header"]["itemsize"],
            frames["frames"],
        )
    ):
        print(f"Frame {idx}: {format_str} {itemsize}")
        if format_str == "=c" and itemsize == 1:
            print(textwrap.indent('"' + frame.tobytes().decode("utf-8") + '"', "    "))
        else:
            print(textwrap.indent(str(frame), "    "))


if __name__ == "__main__":
    inspect_frames()
