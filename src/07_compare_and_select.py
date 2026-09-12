"""
Step 7: Combine every model's test-set metrics, pick the winner, and stage it for
deployment.

Selection rule: highest test R^2, full stop -- no thumb on the scale for the hybrid
model. If a tree ensemble wins, the writeup says so plainly (small tabular datasets
like this one, ~800 training rows, are a well-known hard case for deep learning to
beat gradient-boosted/random-forest ensembles on).

This is also where every file the DEPLOYED APP needs at runtime gets staged into
models/ -- the only data directory that's actually tracked in git (data/interim/
and data/processed/ are gitignored as regeneratable pipeline intermediates). The
app must never read directly from data/processed/ or data/interim/: those don't
exist on a fresh clone (e.g. Streamlit Cloud), only whatever main() writes below.

Output:
  reports/model_comparison.csv        -- every model, sorted by test R^2
  models/final_model.*                -- copy of the winning model, deployment-ready
  models/final_model_meta.json        -- which model won, its metrics, its "kind"
  models/feature_list.json            -- app-facing copy of the 25 feature names
  models/sample_outcomes.csv          -- app-facing copy of just the 2 columns the
                                          app's charts need (target + AI breadth),
                                          not the full 25-feature modeling table
"""
import json
import shutil
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

# maps the "model" name used in metrics_*.csv to (artifact filename, kind, feature_space, target_scaled)
# feature_space is "scaled" (imputed + StandardScaler) or "imputed" (imputed only, raw units)
# target_scaled is True for the Keras models, which are trained on a StandardScaler'd
# target (models/target_scaler.joblib) and need predictions inverse-transformed
MODEL_ARTIFACTS = {
    "Mean baseline": ("mean_baseline.joblib", "sklearn", "scaled", False),
    "Linear Regression": ("linear_regression.joblib", "sklearn", "scaled", False),
    "Stacking (RF+GB+ET -> Ridge)": ("stacking.joblib", "sklearn", "imputed", False),
    "Random Forest (tuned)": ("random_forest.joblib", "sklearn", "imputed", False),
    "Gradient Boosting (tuned)": ("gradient_boosting.joblib", "sklearn", "imputed", False),
    "Extra Trees (tuned)": ("extra_trees.joblib", "sklearn", "imputed", False),
    "Plain NN": ("plain_nn.keras", "keras", "scaled", True),
    "Hybrid Wide & Deep": ("hybrid_wide_deep.keras", "keras", "scaled", True),
}


def _match_artifact(model_name: str):
    if model_name in MODEL_ARTIFACTS:
        return MODEL_ARTIFACTS[model_name]
    # Ridge/Lasso names carry their fitted alpha, e.g. "Ridge (alpha=10)"
    if model_name.startswith("Ridge"):
        return ("ridge.joblib", "sklearn", "scaled", False)
    if model_name.startswith("Lasso"):
        return ("lasso.joblib", "sklearn", "scaled", False)
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
    artifact_name, kind, feature_space, target_scaled = _match_artifact(best_name)

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
        "target_scaled": target_scaled,
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

    # --- Stage app-facing runtime data into models/ (tracked in git), since
    # data/processed/ is gitignored and won't exist on a fresh clone/deploy ---
    shutil.copy2("data/processed/feature_list.json", MODELS_DIR / "feature_list.json")
    print(f"App feature list -> {MODELS_DIR / 'feature_list.json'}")

    model_data = pd.read_csv("data/processed/model_data.csv")
    sample_outcomes = model_data[["ai_f2a2", "ai_breadth_score"]]
    sample_outcomes.to_csv(MODELS_DIR / "sample_outcomes.csv", index=False)
    print(f"App sample outcomes ({len(sample_outcomes)} rows) -> {MODELS_DIR / 'sample_outcomes.csv'}")


if __name__ == "__main__":
    main()
