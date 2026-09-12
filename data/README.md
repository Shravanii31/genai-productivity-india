# Data

This project uses a firm-level survey on AI adoption among Indian firms
(~1,355 firms, 132 raw columns).

## `raw_survey.dta` (not included in this repository)

The raw Stata file is **restricted survey microdata** and is not redistributable.
It is excluded via `.gitignore`. To reproduce this project:

1. Obtain the raw dataset from the original data provider under the applicable
   data-use agreement.
2. Place it at `data/raw_survey.dta`.
3. Run the pipeline scripts in `src/` in numeric order (see the top-level
   `README.md` for details).

## Derived data

`src/01_load_and_clean.py` and `src/02_feature_engineering.py` write derived,
de-identified, feature-engineered tables to `data/interim/` and
`data/processed/`. These are also excluded from version control (see
`.gitignore`) since they are fully reproducible from the raw file and the
pipeline scripts, and because they are still derived from restricted
microdata. Re-run the pipeline to regenerate them.
