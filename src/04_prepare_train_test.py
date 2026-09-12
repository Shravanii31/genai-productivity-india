"""
Step 4: Fixed train/test split and preprocessing, shared by every model downstream.

Everything after this script reads from these saved artifacts rather than
re-splitting or re-fitting preprocessing, so every model in the comparison table is
evaluated on exactly the same held-out test set with exactly the same preprocessing.

- Split: 80/20, random_state=42, fixed for the whole project.
- Preprocessing: median imputation, then StandardScaler -- both fit on the training
  split only, then applied to train/test alike.
  - Tree models use the imputed-but-unscaled features (scaling doesn't matter for
    trees, and keeping raw units makes feature importances easier to read).
  - Linear and deep-learning models use the imputed-and-scaled features.

Output:
  data/processed/{X_train,X_test}_imputed.csv   (trees)
  data/processed/{X_train,X_test}_scaled.csv     (linear / deep learning)
  data/processed/{y_train,y_test}.csv
  models/imputer.joblib, models/scaler.joblib
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/model_data.csv")
FEATURE_LIST_PATH = Path("data/processed/feature_list.json")
OUT_DIR = Path("data/processed")
MODELS_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(FEATURE_LIST_PATH) as fh:
        spec = json.load(fh)
    features, target = spec["features"], spec["target"]

    df = pd.read_csv(DATA_PATH)
    X, y = df[features], df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=features, index=X_test.index
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp), columns=features, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp), columns=features, index=X_test.index
    )

    X_train_imp.to_csv(OUT_DIR / "X_train_imputed.csv", index=False)
    X_test_imp.to_csv(OUT_DIR / "X_test_imputed.csv", index=False)
    X_train_scaled.to_csv(OUT_DIR / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(OUT_DIR / "X_test_scaled.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    joblib.dump(imputer, MODELS_DIR / "imputer.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Saved imputed/scaled splits to data/processed/, preprocessing to models/")


if __name__ == "__main__":
    main()
