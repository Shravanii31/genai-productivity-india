"""
Step 1: Load the raw Stata survey file and apply cleaning rules.

- Loads data/raw_survey.dta with convert_categoricals=False.
- Converts survey sentinel codes (-9 = "Don't know", -7 = "Does not apply") to NaN
  across all numeric columns. Columns are cast to float64 first and cleaned
  column-by-column with boolean masking, since a blanket df.replace([-9, -7], np.nan)
  throws an IndexError on this dataset (int8-dtype columns).
- Recovers structural missingness: several blocks of AI-adoption follow-up questions
  are only asked to firms that report adopting AI (ai_adopt1 == 1). For non-adopters,
  NaN in these columns is not "missing data" -- it means the question did not apply,
  which in this survey's coding is a legitimate "No" (value 2). We fill those NaNs
  with 2 rather than imputing them later.

Output: data/interim/cleaned.parquet
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw_survey.dta")
OUT_DIR = Path("data/interim")
OUT_PATH = OUT_DIR / "cleaned.parquet"

SENTINELS = [-9, -7]

# Column-name prefixes that are only asked of AI adopters; NaN there means
# "does not apply because this firm is not an AI adopter" -> legitimate "No" (2).
STRUCTURAL_MISSING_PATTERNS = [
    r"^ai_b3a[1-8]$",
    r"^ai_b4a[1-7]$",
    r"^ai_b6a[1-5]$",
    r"^ai_b6b",
    r"^ai_b6c",
    r"^ai_b6d",
    r"^ai_c1a[1-4]$",
    r"^ai_c1f$",
    r"^ai_c1h$",
    r"^ai_b2b[1-3]$",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_stata(path, convert_categoricals=False)
    return df


def clean_sentinels(df: pd.DataFrame, sentinels=SENTINELS) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = df[c].astype("float64")
        mask = df[c].isin(sentinels)
        df.loc[mask, c] = np.nan
    return df


def recover_structural_missingness(
    df: pd.DataFrame, patterns=STRUCTURAL_MISSING_PATTERNS
) -> pd.DataFrame:
    df = df.copy()
    compiled = [re.compile(p) for p in patterns]
    target_cols = [
        c for c in df.columns if any(p.match(c) for p in compiled)
    ]
    for c in target_cols:
        df[c] = df[c].fillna(2)
    return df, target_cols


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw()
    print(f"Raw shape: {df.shape}")

    df = clean_sentinels(df)
    print("Sentinel codes (-9, -7) converted to NaN.")

    df, recovered_cols = recover_structural_missingness(df)
    print(
        f"Structural missingness recovered (NaN -> 2) for "
        f"{len(recovered_cols)} adopter-only columns."
    )

    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote cleaned data to {OUT_PATH}  shape={df.shape}")


if __name__ == "__main__":
    main()
