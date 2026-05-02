import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import pickle
import os
from datetime import datetime

def generate_data(n_samples=1000, drift=False):
    if drift:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            random_state=99,
            flip_y=0.3
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            random_state=42
        )
    columns = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df

def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("mlops-pipeline")

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        }

        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("Training metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        X_train.to_csv("data/reference_data.csv", index=False)
        X_test.to_csv("data/current_data.csv", index=False)
        print("Model saved to models/model.pkl")

        return model, metrics

if __name__ == "__main__":
    print("Generating training data...")
    df = generate_data(n_samples=1000)
    print("Training model...")
    model, metrics = train_model(df)
    print("Done!")
