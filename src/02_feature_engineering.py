"""
Step 2: Feature engineering.

Builds the full engineered feature table (55+ columns, kept around for EDA / the
"Data Understanding" story) from the cleaned survey data, then selects the
validated 25-feature subset used for modeling.

Target: ai_f2a2 ("Expected productivity increase at current AI pace, %").
Modeling sample = rows where the target is non-null (~986 of 1355 firms).

Output:
  data/processed/features_full.csv   -- full engineered table + target, all 1355 rows
  data/processed/model_data.csv      -- 25 selected features + target, modeling sample only
  data/processed/feature_list.json   -- the 25 modeling feature names, in order
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/interim/cleaned.parquet")
OUT_DIR = Path("data/processed")

TARGET = "ai_f2a2"

AI_TECH_COLS = [f"ai_b2a{i}" for i in range(1, 7)]        # ML, chatbot, agent, genAI, automation, autonomous
FUNCTIONAL_COLS = [f"ai_b4a{i}" for i in range(1, 8)]       # R&D, marketing, supply chain, admin, strategy, compliance, QC
TASK_COLS = [f"ai_b3a{i}" for i in range(1, 9)]             # 8 task-level AI use flags
POLICY_COLS = ["ai_g1a", "ai_g1b", "ai_g1c", "ai_g1d", "ai_g1e", "ai_g1f"]

FINAL_25_FEATURES = [
    "reskilling_need_share",
    "workers_wanting_more_hours_share",
    "idle_time_share",
    "admin_worker_share",
    "share_clerical",
    "sales_seasonality_ratio",
    "policy_support_score",
    "ai_b3a3_flag",
    "share_professionals",
    "ai_b2a4_flag",
    "share_managers",
    "worker_seasonality_ratio",
    "uses_genai",
    "share_computer_use",
    "log_workers",
    "task_breadth",
    "finance_digitization",
    "ai_g1b_flag",
    "customer_digitization",
    "ai_b3a2_flag",
    "can_absorb_demand_shock",
    "supplychain_digitization",
    "ai_breadth_score",
    "ai_g1f_flag",
    "has_website",
]


def _digitization_ladder(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Ordinal 1..N ladder: highest-numbered stage flagged 'yes' (==1)."""
    cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.replace(prefix, "")),
    )
    stage = pd.Series(0, index=df.index, dtype="float64")
    for i, c in enumerate(cols, start=1):
        is_yes = df[c] == 1
        stage = np.where(is_yes, i, stage)
        stage = pd.Series(stage, index=df.index)
    return stage


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # --- firm profile ---
    f["log_workers"] = np.log1p(df["e1a"])
    f["share_managers"] = df["ai_a2a"] / df["e1a"]
    f["share_professionals"] = df["ai_a2b"] / df["e1a"]
    f["share_clerical"] = df["ai_a2c"] / df["e1a"]
    f["admin_worker_share"] = df["i1a"] / df["e1a"]
    f["sector_ai_intensive"] = (
        df["stratificationsectorname"] == "AI intensive"
    ).astype(int)

    # --- AI adoption / usage ---
    f["has_website"] = (df["ai_b1a3"] == 1).astype(int)
    f["share_computer_use"] = df["ai_b1b"]
    f["ai_adopter"] = df["ai_adopt1"]

    for c in AI_TECH_COLS:
        f[f"{c}_flag"] = (df[c] == 1).astype(int)
    f["ai_breadth_score"] = f[[f"{c}_flag" for c in AI_TECH_COLS]].sum(axis=1)
    f["uses_genai"] = f["ai_b2a4_flag"]

    for c in FUNCTIONAL_COLS:
        f[f"{c}_flag"] = (df[c] == 1).astype(int)
    f["functional_breadth"] = f[[f"{c}_flag" for c in FUNCTIONAL_COLS]].sum(axis=1)

    for c in TASK_COLS:
        f[f"{c}_flag"] = (df[c] == 1).astype(int)
    f["task_breadth"] = f[[f"{c}_flag" for c in TASK_COLS]].sum(axis=1)
    # ai_b3a2_flag (summarizing), ai_b3a3_flag (drafting) already present individually

    # --- digitization maturity ladders ---
    f["finance_digitization"] = _digitization_ladder(df, "fatb7a")
    f["supplychain_digitization"] = _digitization_ladder(df, "fatb8a")
    f["customer_digitization"] = _digitization_ladder(df, "fat_b9a")

    # --- workforce impact ---
    f["hired_due_to_ai"] = (df["ai_c1a1"] == 1).astype(int)
    f["laidoff_due_to_ai"] = (df["ai_c1a2"] == 1).astype(int)
    f["froze_hiring_due_to_ai"] = (df["ai_c1a3"] == 1).astype(int)
    f["reassigned_due_to_ai"] = (df["ai_c1a4"] == 1).astype(int)
    f["gave_ai_training"] = (df["ai_c1f"] == 1).astype(int)
    f["tried_hiring_ai_talent"] = (df["ai_c1h"] == 1).astype(int)

    # --- policy support ---
    for c in POLICY_COLS:
        f[f"{c}_flag"] = (df[c] == 1).astype(int)
    f["policy_support_score"] = f[[f"{c}_flag" for c in POLICY_COLS]].sum(axis=1)

    # --- operating conditions ---
    f["idle_time_share"] = df["jobs_h1"]
    f["workers_wanting_more_hours_share"] = df["jobs_h3"]
    f["can_absorb_demand_shock"] = (df["jobs_h4"] == 1).astype(int)
    f["capacity_utilization"] = df["jobs_h5"]
    f["reskilling_need_share"] = df["ai_f1b"]

    # seasonality ratios -- clip to 20 to remove a data-entry outlier that otherwise
    # produces ratios of ~140,000x (a near-zero denominator on one firm) and
    # destabilizes gradient-based models
    f["sales_seasonality_ratio"] = (df["jobs_h6a"] / df["jobs_h7a"]).clip(upper=20)
    f["worker_seasonality_ratio"] = (df["jobs_h6b"] / df["jobs_h7b"]).clip(upper=20)

    f["expects_employee_increase"] = (df["ai_f1a1"] == 2).astype(int)

    f[TARGET] = df[TARGET]

    return f


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(IN_PATH)

    full = build_features(df)
    full.to_csv(OUT_DIR / "features_full.csv", index=False)
    print(f"Full engineered table: {full.shape} -> {OUT_DIR / 'features_full.csv'}")

    model_df = full.loc[full[TARGET].notna(), FINAL_25_FEATURES + [TARGET]].copy()
    model_df.to_csv(OUT_DIR / "model_data.csv", index=False)
    print(f"Modeling table (25 features): {model_df.shape} -> {OUT_DIR / 'model_data.csv'}")

    with open(OUT_DIR / "feature_list.json", "w") as fh:
        json.dump({"features": FINAL_25_FEATURES, "target": TARGET}, fh, indent=2)
    print(f"Feature list -> {OUT_DIR / 'feature_list.json'}")

    print("\nMissingness in modeling table (top 10):")
    print(model_df.isna().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
