"""
Step 5: Train and evaluate the non-hybrid baseline models.

Mean baseline, Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting,
Extra Trees (each tuned with RandomizedSearchCV, 3-fold CV, on the training set
only), and a Stacking ensemble of the three tuned tree models with a Ridge
meta-learner.

All models are evaluated on the same held-out test set produced by
04_prepare_train_test.py. Tree-based models use imputed-but-unscaled features;
linear models use imputed-and-scaled features.

Output:
  reports/metrics_baselines.csv
  models/{model_name}.joblib for every model trained here
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42


def load_splits():
    X_train_imp = pd.read_csv(DATA_DIR / "X_train_imputed.csv")
    X_test_imp = pd.read_csv(DATA_DIR / "X_test_imputed.csv")
    X_train_scaled = pd.read_csv(DATA_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(DATA_DIR / "X_test_scaled.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]
    return X_train_imp, X_test_imp, X_train_scaled, X_test_scaled, y_train, y_test


def evaluate(name, model, X_test, y_test, results, fit_seconds=None):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    results.append(
        {"model": name, "r2": r2, "rmse": rmse, "mae": mae, "fit_seconds": fit_seconds}
    )
    print(f"{name:28s}  R2={r2:7.4f}  RMSE={rmse:7.4f}  MAE={mae:7.4f}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_imp, X_test_imp, X_train_scaled, X_test_scaled, y_train, y_test = load_splits()
    results = []

    # --- Mean baseline ---
    t0 = time.time()
    dummy = DummyRegressor(strategy="mean").fit(X_train_scaled, y_train)
    evaluate("Mean baseline", dummy, X_test_scaled, y_test, results, time.time() - t0)
    joblib.dump(dummy, MODELS_DIR / "mean_baseline.joblib")

    # --- Plain Linear Regression ---
    t0 = time.time()
    lin = LinearRegression().fit(X_train_scaled, y_train)
    evaluate("Linear Regression", lin, X_test_scaled, y_test, results, time.time() - t0)
    joblib.dump(lin, MODELS_DIR / "linear_regression.joblib")

    # --- Ridge (alpha via CV) ---
    t0 = time.time()
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5).fit(X_train_scaled, y_train)
    evaluate(f"Ridge (alpha={ridge.alpha_:.4g})", ridge, X_test_scaled, y_test, results, time.time() - t0)
    joblib.dump(ridge, MODELS_DIR / "ridge.joblib")

    # --- Lasso (alpha via CV) ---
    t0 = time.time()
    lasso = LassoCV(alphas=np.logspace(-3, 1, 25), cv=5, random_state=RANDOM_STATE, max_iter=20000).fit(
        X_train_scaled, y_train
    )
    evaluate(f"Lasso (alpha={lasso.alpha_:.4g})", lasso, X_test_scaled, y_test, results, time.time() - t0)
    joblib.dump(lasso, MODELS_DIR / "lasso.joblib")

    # --- Random Forest (RandomizedSearchCV) ---
    t0 = time.time()
    rf_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        rf_grid, n_iter=30, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train_imp, y_train)
    rf_best = rf_search.best_estimator_
    evaluate("Random Forest (tuned)", rf_best, X_test_imp, y_test, results, time.time() - t0)
    joblib.dump(rf_best, MODELS_DIR / "random_forest.joblib")

    # --- Gradient Boosting (RandomizedSearchCV) ---
    t0 = time.time()
    gb_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_leaf": [1, 2, 4],
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        gb_grid, n_iter=30, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train_imp, y_train)
    gb_best = gb_search.best_estimator_
    evaluate("Gradient Boosting (tuned)", gb_best, X_test_imp, y_test, results, time.time() - t0)
    joblib.dump(gb_best, MODELS_DIR / "gradient_boosting.joblib")

    # --- Extra Trees (RandomizedSearchCV) ---
    t0 = time.time()
    et_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    et_search = RandomizedSearchCV(
        ExtraTreesRegressor(random_state=RANDOM_STATE),
        et_grid, n_iter=30, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train_imp, y_train)
    et_best = et_search.best_estimator_
    evaluate("Extra Trees (tuned)", et_best, X_test_imp, y_test, results, time.time() - t0)
    joblib.dump(et_best, MODELS_DIR / "extra_trees.joblib")

    # --- Stacking ensemble (RF + GB + ET -> Ridge meta-learner) ---
    t0 = time.time()
    stack = StackingRegressor(
        estimators=[("rf", rf_best), ("gb", gb_best), ("et", et_best)],
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1,
    ).fit(X_train_imp, y_train)
    evaluate("Stacking (RF+GB+ET -> Ridge)", stack, X_test_imp, y_test, results, time.time() - t0)
    joblib.dump(stack, MODELS_DIR / "stacking.joblib")

    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    results_df.to_csv(REPORTS_DIR / "metrics_baselines.csv", index=False)
    print(f"\nSaved -> {REPORTS_DIR / 'metrics_baselines.csv'}")
    print(results_df.to_string(index=False))

    best_params = {
        "random_forest": rf_search.best_params_,
        "gradient_boosting": gb_search.best_params_,
        "extra_trees": et_search.best_params_,
        "ridge_alpha": ridge.alpha_,
        "lasso_alpha": lasso.alpha_,
    }
    with open(REPORTS_DIR / "baseline_best_params.json", "w") as fh:
        json.dump(best_params, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
