"""
Step 7: Combine every model's test-set metrics, pick the winner, and stage it for
deployment.

Selection rule: highest test R^2, full stop -- no thumb on the scale for the hybrid
model. If a tree ensemble wins, the writeup says so plainly (small tabular datasets
like this one, ~800 training rows, are a well-known hard case for deep learning to
beat gradient-boosted/random-forest ensembles on).

Output:
  reports/model_comparison.csv        -- every model, sorted by test R^2
  models/final_model.*                -- copy of the winning model, deployment-ready
  models/final_model_meta.json        -- which model won, its metrics, its "kind"
"""
import json
import shutil
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

# maps the "model" name used in metrics_*.csv to (artifact filename, kind, feature_space)
# feature_space is "scaled" (imputed + StandardScaler) or "imputed" (imputed only, raw units)
MODEL_ARTIFACTS = {
    "Mean baseline": ("mean_baseline.joblib", "sklearn", "scaled"),
    "Linear Regression": ("linear_regression.joblib", "sklearn", "scaled"),
    "Stacking (RF+GB+ET -> Ridge)": ("stacking.joblib", "sklearn", "imputed"),
    "Random Forest (tuned)": ("random_forest.joblib", "sklearn", "imputed"),
    "Gradient Boosting (tuned)": ("gradient_boosting.joblib", "sklearn", "imputed"),
    "Extra Trees (tuned)": ("extra_trees.joblib", "sklearn", "imputed"),
    "Plain NN": ("plain_nn.keras", "keras", "scaled"),
    "Hybrid Wide & Deep": ("hybrid_wide_deep.keras", "keras", "scaled"),
}


def _match_artifact(model_name: str):
    if model_name in MODEL_ARTIFACTS:
        return MODEL_ARTIFACTS[model_name]
    # Ridge/Lasso names carry their fitted alpha, e.g. "Ridge (alpha=10)"
    if model_name.startswith("Ridge"):
        return ("ridge.joblib", "sklearn", "scaled")
    if model_name.startswith("Lasso"):
        return ("lasso.joblib", "sklearn", "scaled")
    raise KeyError(f"No artifact mapping for model '{model_name}'")


def main():
    baselines = pd.read_csv(REPORTS_DIR / "metrics_baselines.csv")
    deep = pd.read_csv(REPORTS_DIR / "metrics_deep.csv")
    combined = pd.concat([baselines, deep], ignore_index=True).sort_values(
        "r2", ascending=False
    )
    combined.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print("Full model comparison (sorted by test R^2):")
    print(combined.to_string(index=False))

    best_row = combined.iloc[0]
    best_name = best_row["model"]
    artifact_name, kind, feature_space = _match_artifact(best_name)

    src = MODELS_DIR / artifact_name
    dst = MODELS_DIR / ("final_model.keras" if kind == "keras" else "final_model.joblib")

    if kind == "keras":
        shutil.copytree(src, dst, dirs_exist_ok=True) if src.is_dir() else shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

    meta = {
        "model_name": best_name,
        "artifact": dst.name,
        "kind": kind,
        "feature_space": feature_space,
        "r2": float(best_row["r2"]),
        "rmse": float(best_row["rmse"]),
        "mae": float(best_row["mae"]),
        "n_test": None,
    }
    y_test = pd.read_csv("data/processed/y_test.csv")
    meta["n_test"] = int(len(y_test))
    y_train = pd.read_csv("data/processed/y_train.csv")
    meta["n_train"] = int(len(y_train))

    with open(MODELS_DIR / "final_model_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nWinner: {best_name}  (R2={meta['r2']:.4f})")
    print(f"Deployed as {dst} ; metadata -> {MODELS_DIR / 'final_model_meta.json'}")


if __name__ == "__main__":
    main()
