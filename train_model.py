"""
Model Training Script
---------------------
Trains a RandomForestClassifier on the windowed features produced by
extract_features.py, evaluates it, and saves the trained model to disk.

Usage:
    python train_model.py --features features.csv --model_out model.pkl
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model_out", default="model.pkl")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.features)

    drop_cols = [c for c in ["label", "session_file"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"]

    print("Class distribution:")
    print(y.value_counts())
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    print()

    print("=== Feature Importances ===")
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False))

    joblib.dump({"model": clf, "feature_columns": list(X.columns)}, args.model_out)
    print(f"\nSaved trained model to {args.model_out}")


if __name__ == "__main__":
    main()
