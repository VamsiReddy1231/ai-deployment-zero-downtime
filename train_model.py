from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save(path="model.joblib"):
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, path)
    print("Model saved as", path)

if __name__ == "__main__":
    train_and_save()
