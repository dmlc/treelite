import pickle
import argparse
import numpy as np
import treelite
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def save(args):
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=10)
    clf.fit(X, y)
    with open(args.model_pickle_path, "wb") as f:
        pickle.dump(clf, f)
    tl_model = treelite.sklearn.import_model(clf)
    tl_model.serialize(args.checkpoint_path)


def load(args):
    X, y = load_iris(return_X_y=True)
    with open(args.model_pickle_path, "rb") as f:
        clf = pickle.load(f)
    tl_model = treelite.Model.deserialize(args.checkpoint_path)
    expected_prob = clf.predict_proba(X)
    out_prob = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)
    print("Test passed!")


def main(args):
    if treelite.__version__ != args.expected_treelite_version:
        raise ValueError(f"Expected Treelite {args.expected_treelite_version} "
                         f"but running Treelite {treelite.__version__}")
    if args.task == "save":
        save(args)
    elif args.task == "load":
        load(args)
    else:
        raise ValueError(f"Unknown task type: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["save", "load"],
                        required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-pickle-path", type=str, required=True)
    parser.add_argument("--expected-treelite-version", type=str, required=True)
    args = parser.parse_args()
    main(args)
