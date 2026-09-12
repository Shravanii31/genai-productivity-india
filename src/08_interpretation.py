"""
Step 8: Interpret the final deployed model (whichever one won in step 7).

- Manual permutation importance on the test set: shuffle one column at a time,
  measure the drop in R^2. Written by hand (rather than
  sklearn.inspection.permutation_importance) so it works identically whether the
  final model is a scikit-learn estimator or a Keras model, which
  permutation_importance can't call directly.
- Diagnostic plots: predicted-vs-actual scatter, residual plot.
- reports/model_interpretation.md: predictors, practical usefulness, limitations,
  possible improvements.

Output:
  reports/permutation_importance.csv
  figures/08_pred_vs_actual.png, figures/09_residuals.png, figures/10_permutation_importance.png
  reports/model_interpretation.md
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIG_DIR = Path("figures")
RANDOM_STATE = 42


def load_final_model():
    with open(MODELS_DIR / "final_model_meta.json") as fh:
        meta = json.load(fh)
    if meta["kind"] == "keras":
        from tensorflow import keras
        model = keras.models.load_model(MODELS_DIR / meta["artifact"])
        predict_fn = lambda X: model.predict(X, verbose=0).flatten()
    else:
        model = joblib.load(MODELS_DIR / meta["artifact"])
        predict_fn = lambda X: model.predict(X)
    return meta, predict_fn


def load_test_data(feature_space: str):
    suffix = "scaled" if feature_space == "scaled" else "imputed"
    X_test = pd.read_csv(DATA_DIR / f"X_test_{suffix}.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]
    return X_test, y_test


def permutation_importance(predict_fn, X_test, y_test, n_repeats=15, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    baseline_r2 = r2_score(y_test, predict_fn(X_test.values))
    importances = []
    for col in X_test.columns:
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].values)
            shuffled_r2 = r2_score(y_test, predict_fn(X_shuffled.values))
            drops.append(baseline_r2 - shuffled_r2)
        importances.append({"feature": col, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return baseline_r2, pd.DataFrame(importances).sort_values("importance_mean", ascending=False)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meta, predict_fn = load_final_model()
    X_test, y_test = load_test_data(meta["feature_space"])

    print(f"Final model: {meta['model_name']}  (test R2={meta['r2']:.4f})")

    baseline_r2, importance_df = permutation_importance(predict_fn, X_test, y_test)
    importance_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    print("\nTop 10 predictors by permutation importance:")
    print(importance_df.head(10).to_string(index=False))

    # Diagnostic plot 1: predicted vs actual
    pred = predict_fn(X_test.values)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.5, color="#33C7B0", edgecolor="none")
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, color="#F0A83A", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual expected productivity increase (%)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual -- {meta['model_name']} (R2={baseline_r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # Diagnostic plot 2: residuals
    residuals = y_test.values - pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(pred, residuals, alpha=0.5, color="#33C7B0", edgecolor="none")
    axes[0].axhline(0, color="#F0A83A", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=25, color="#F0A83A")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_residuals.png", dpi=150)
    plt.close(fig)

    # Permutation importance plot (top 15)
    top15 = importance_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top15["feature"], top15["importance_mean"], xerr=top15["importance_std"], color="#33C7B0")
    ax.set_xlabel("Mean drop in R^2 when shuffled")
    ax.set_title(f"Permutation Importance -- {meta['model_name']}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_permutation_importance.png", dpi=150)
    plt.close(fig)

    write_report(meta, baseline_r2, importance_df, residuals)
    print(f"\nWrote {REPORTS_DIR / 'model_interpretation.md'}")


def write_report(meta, baseline_r2, importance_df, residuals):
    top10 = importance_df.head(10)
    n_train = meta.get("n_train")
    n_test = meta.get("n_test")

    lines = []
    lines.append("# Model Interpretation\n")
    lines.append(f"**Final deployed model:** {meta['model_name']}\n")
    lines.append(
        f"**Test-set performance:** R2 = {meta['r2']:.4f}, RMSE = {meta['rmse']:.4f}, "
        f"MAE = {meta['mae']:.4f}  (n_train = {n_train}, n_test = {n_test})\n"
    )

    lines.append("## Important predictors\n")
    lines.append(
        "Ranked by permutation importance on the held-out test set (mean drop in R^2 "
        "over 15 shuffles per feature):\n"
    )
    for _, row in top10.iterrows():
        lines.append(f"- **{row['feature']}**: {row['importance_mean']:.4f} (+/- {row['importance_std']:.4f})")
    lines.append("")

    lines.append("## Practical usefulness\n")
    lines.append(
        "The model surfaces which firm characteristics associate with a firm's own "
        "expectation of AI-driven productivity gains: breadth and maturity of AI/digital "
        "adoption (task breadth, generative-AI use, digitization ladders), workforce "
        "composition (share of managers/professionals/clerical staff, firm size), and "
        "operating slack (idle time, seasonality, capacity headroom). This is useful as a "
        "directional screening tool -- e.g. for a policy team or lender trying to identify "
        "which firms report the most AI-driven upside -- rather than as a precise forecast "
        "for any single firm.\n"
    )

    lines.append("## Limitations\n")
    lines.append(
        "- The target (`ai_f2a2`) is a **subjective, self-reported expectation**, not a "
        "measured outcome -- it captures what a firm's respondent believes will happen, "
        "which is shaped by optimism, awareness of AI, and expectations rather than "
        "realized productivity data.\n"
        f"- R^2 of {meta['r2']:.2f} is a strong, honest result for this kind of "
        "noisy, self-reported survey target; an R^2 much higher than ~0.65 here would be a "
        "signal of leakage rather than genuine predictive skill.\n"
        "- The modeling sample is limited to the ~986 firms that answered the target "
        "question (of 1,355 surveyed), and training used only ~80% of that "
        f"({n_train} firms) -- a small-N regime where estimates carry real uncertainty.\n"
        "- Cross-sectional survey data: associations here are not causal claims.\n"
    )

    lines.append("## Possible improvements\n")
    lines.append(
        "- Collect a longitudinal follow-up (realized productivity change vs the prior "
        "expectation) to validate or recalibrate the self-reported target.\n"
        "- Expand the training sample -- both models compared here (tree ensembles and the "
        "wide & deep hybrid) are constrained more by sample size than by architecture.\n"
        "- Add sector/industry fixed effects or interaction terms between AI adoption "
        "breadth and firm size, which permutation importance suggests may be under-captured "
        "by the current linear + tree feature set.\n"
    )

    with open(REPORTS_DIR / "model_interpretation.md", "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
