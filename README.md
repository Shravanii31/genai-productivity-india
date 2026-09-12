# AI Adoption & Expected Productivity Gains -- India Firm Survey

A regression project predicting a firm's **self-reported expected productivity
increase from AI** using a firm-level survey of Indian businesses, comparing a
hybrid Wide & Deep (linear + neural network) model against classical and
ensemble baselines, and deploying the best-performing model as a Streamlit app.

## Problem statement

Given firm characteristics -- size, workforce composition, AI adoption breadth,
digitization maturity, operating conditions, and policy support -- predict
`ai_f2a2`, the percentage productivity increase a firm expects from AI at its
current pace of adoption.

## Objectives

1. Build a clean, reproducible pipeline from raw restricted survey microdata to
   a modeling-ready feature table.
2. Engineer a validated set of 25 predictive features from 132 raw survey columns.
3. Build a **hybrid Wide & Deep model** combining linear regression and a deep
   neural network in one architecture, and honestly benchmark it against linear,
   regularized, and tree-ensemble baselines.
4. Interpret the winning model (permutation importance, diagnostics).
5. Deploy the winning model behind a Streamlit wizard app.

## Dataset

~1,355 Indian firms, 132 survey columns, restricted microdata (see
[data/README.md](data/README.md) -- the raw file is not included in this repo).
The modeling sample is the ~986 firms with a non-missing target.

## Repository structure

```
├── src/                       # numbered pipeline scripts, run in order
│   ├── 01_load_and_clean.py
│   ├── 02_feature_engineering.py
│   ├── 03_eda.py
│   ├── 04_prepare_train_test.py
│   ├── 05_train_baseline_models.py
│   ├── 06_train_deep_models.py
│   ├── 07_compare_and_select.py
│   └── 08_interpretation.py
├── app/app.py                 # Streamlit deployment app
├── models/                    # trained model artifacts + final_model_meta.json
├── reports/                   # metrics, comparison table, model_interpretation.md
├── figures/                   # EDA + diagnostic plots
├── data/README.md             # raw data is excluded -- restricted microdata
├── .streamlit/config.toml     # app theme
├── requirements.txt           # pinned deployment dependencies
├── requirements-dev.txt       # adds dev/training-only dependencies
└── .gitignore
```

## How to run

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# place the raw file first: data/raw_survey.dta
python src/01_load_and_clean.py
python src/02_feature_engineering.py
python src/03_eda.py
python src/04_prepare_train_test.py
python src/05_train_baseline_models.py
python src/06_train_deep_models.py
python src/07_compare_and_select.py
python src/08_interpretation.py

streamlit run app/app.py
```

## Methodology

### Data pre-processing
- Sentinel codes -9 ("Don't know") and -7 ("Does not apply") converted to NaN.
- Structural missingness recovery: adopter-only follow-up questions are filled
  with "No" (2) for non-adopters rather than imputed, since the missingness there
  is not random -- it reflects survey skip logic.
- Median imputation + StandardScaler, both fit on the training split only.

### Feature engineering
55+ engineered columns built from the raw survey (firm profile, AI adoption
breadth, functional/task-level AI use, digitization maturity ladders, workforce
impact, policy support, operating conditions), of which a validated 25-feature
subset (selected via prior Random Forest importance analysis) is used for
modeling. See `src/02_feature_engineering.py` for exact definitions.

### Model development
A **Wide & Deep** architecture (Cheng et al., 2016), implemented with the Keras
functional API:
- **Wide component**: `Dense(1, linear)` applied directly to the scaled inputs --
  mathematically equivalent to linear regression, capturing direct
  feature-target relationships.
- **Deep component**: 3 hidden layers (128 -> 64 -> 32, ReLU) with
  BatchNormalization + Dropout (0.3 / 0.2 / 0.1), capturing nonlinear
  interactions.
- The two outputs are summed into the final prediction, trained jointly with one
  Adam optimizer against one MSE loss -- not two models stacked after the fact.

This was chosen to satisfy the assignment's ML+DL combination requirement while
keeping the linear signal directly interpretable via the wide path's weights.
It's benchmarked against: a mean baseline, plain Linear Regression, Ridge, Lasso,
Random Forest, Gradient Boosting, Extra Trees (all three tree models tuned via
`RandomizedSearchCV`, 3-fold CV), a Stacking ensemble (RF+GB+ET -> Ridge
meta-learner), and a plain feedforward NN (the hybrid's deep component alone, for
a controlled comparison).

### Model validation
Fixed 80/20 train-test split (`random_state=42`) used identically across every
model. Hyperparameter tuning used 3-5 fold CV on the training set only, never
touching the test set. Metrics: R^2, RMSE, MAE, all on the same held-out test set.

## Results

See `reports/model_comparison.csv` for the full table across all models
(regenerated by `src/07_compare_and_select.py`).

## Model interpretation

See `reports/model_interpretation.md` for permutation importance, diagnostic
plots, and an honest discussion of limitations (the target is a subjective,
self-reported expectation, not a measured outcome).

## Limitations & future scope

See `reports/model_interpretation.md`.

## References

- Cheng, H.-T., et al. (2016). *Wide & Deep Learning for Recommender Systems.*
  arXiv:1606.07792.

## Appendix

Full engineered feature table, EDA figures, and best hyperparameters are in
`figures/` and `reports/`.
