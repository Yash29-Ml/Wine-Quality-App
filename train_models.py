import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pickle
import os

def prepare_and_train_models(data_path="winequality-red.csv", sep=";", model_dir="models"):
    df = pd.read_csv(data_path, sep=sep)

    # Sanity check
    assert "quality" in df.columns, "❌ 'quality' column not found. Check delimiter or file format."

    # Create classification labels
    def quality_to_class(q):
        if q <= 5:
            return "Low"
        elif q == 6:
            return "Medium"
        else:
            return "High"

    X = df.drop("quality", axis=1)
    y = df["quality"]
    y_class = df["quality"].apply(quality_to_class)

    # Split once for both targets
    X_train, _, y_train, _, y_class_train, _ = train_test_split(
        X, y, y_class, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_scaled, y_class_train)

    # Train regressor
    reg = KNeighborsRegressor(n_neighbors=5)
    reg.fit(X_train_scaled, y_train)

    # Save models
    os.makedirs(model_dir, exist_ok=True)
    pickle.dump(clf, open(f"{model_dir}/knn_classifier.pkl", "wb"))
    pickle.dump(reg, open(f"{model_dir}/knn_regressor.pkl", "wb"))
    pickle.dump(scaler, open(f"{model_dir}/scaler.pkl", "wb"))

    print(" Models and scaler saved.")

if __name__ == "__main__":
    prepare_and_train_models()