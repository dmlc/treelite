"""Test for serialization, via buffer protocol"""
import treelite
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from serializer import treelite_serialize, treelite_deserialize

def test_serialize_as_buffer():
    """Test whether Treelite objects can be serialized to a buffer"""
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)

    tl_model = treelite.sklearn.import_model(clf)
    frames = treelite_serialize(tl_model)
    # This call should succeed
    tl_model2 = treelite_deserialize(frames)
    # This call should succeed too

if __name__ == '__main__':
    test_serialize_as_buffer()
