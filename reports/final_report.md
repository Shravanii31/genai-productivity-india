# Predicting Expected Productivity Gains from AI Adoption Among Indian Firms

*A regression analysis combining linear and deep-learning models, benchmarked against classical and ensemble baselines*

## 1. Introduction

Artificial intelligence adoption is reshaping firm-level productivity expectations
worldwide, and India's firms are no exception. This project analyzes a firm-level
survey of Indian businesses to understand which firm characteristics -- size,
workforce composition, AI adoption breadth, digitization maturity, operating
conditions, and policy support -- are associated with a firm's own expectation of
how much AI will increase its productivity. The end product is a regression model,
deployed as an interactive web tool, that estimates this expected gain from a
short set of firm inputs.

## 2. Problem Statement

Given a set of firm-level features describing size, AI usage, digitization, and
operating context, predict `ai_f2a2`: the percentage productivity increase a firm
expects from AI at its current pace of adoption. This is a supervised regression
problem on a modestly sized (~986 firm), moderately noisy survey target.

## 3. Objectives

1. Build a reproducible pipeline from raw restricted survey microdata to a clean,
   modeling-ready feature table.
2. Engineer a validated set of predictive features from 132 raw survey columns.
3. Build a **hybrid Wide & Deep model** that combines linear regression and a deep
   neural network in a single architecture, and benchmark it honestly against
   linear, regularized, and tree-ensemble baselines.
4. Select the best-performing model on a held-out test set and interpret it.
5. Deploy the winning model behind an interactive Streamlit application.

## 4. Dataset Description

- **Source**: firm-level AI-adoption survey of Indian businesses.
- **Size**: 1,355 firms, 132 raw survey columns.
- **Modeling sample**: 986 firms with a non-missing target (`ai_f2a2`).
- **Target**: `ai_f2a2` -- expected productivity increase (%) at the firm's
  current pace of AI adoption. Mean ~11%, range 1-65% in the modeling sample.
- **Access**: restricted microdata, not redistributed in this repository (see
  `data/README.md`).

## 5. Exploratory Data Analysis

Key findings (figures in `figures/01`-`05`):

- The target is right-skewed: most firms expect modest gains (5-15%), with a long
  tail of firms expecting much larger gains.
- AI adopters report modestly higher expected productivity gains than
  non-adopters, though the distributions overlap substantially -- adoption status
  alone is a weak predictor.
- Firm size (`log_workers`) shows no strong linear relationship with the target
  once other factors are considered.
- Missingness is concentrated in a handful of operating-condition columns
  (reskilling need, hours-desired share) and is not catastrophic (<10% for most
  modeling features).

## 6. Data Pre-processing

1. **Sentinel cleaning**: survey codes `-9` ("Don't know") and `-7` ("Does not
   apply") were converted to `NaN` across all numeric columns. This was done
   column-by-column with boolean masking after casting to `float64`, since a
   blanket `df.replace([-9, -7], np.nan)` throws an `IndexError` on this
   dataset's `int8`-typed columns.
2. **Structural missingness recovery**: several follow-up question blocks
   (`ai_b3a*`, `ai_b4a*`, `ai_b6*`, `ai_c1a*`, `ai_c1f`, `ai_c1h`, `ai_b2b*`) are
   only asked of firms that report adopting AI. For non-adopters, `NaN` here does
   not mean missing data -- it means the question did not apply, which in this
   survey's coding is a legitimate "No" (value `2`). These were filled with `2`
   rather than imputed.
3. **Feature engineering**: 55+ engineered columns built from raw survey items
   (see Section 7), of which a validated 25-feature subset (selected via prior
   Random Forest importance analysis) was used for modeling.
4. **Seasonality outlier clipping**: `sales_seasonality_ratio` and
   `worker_seasonality_ratio` are peak/off-season ratios; one firm's near-zero
   denominator produced a ratio of roughly 140,000x, a data-entry artifact rather
   than real variation. Both ratios were clipped to a maximum of 20 -- without
   this, gradient-based models destabilize completely (the unclipped hybrid model
   diverged to R^2 = -43,000 in early testing).
5. **Train/test split**: fixed 80/20 split, `random_state=42`, used identically
   across every model (788 train / 198 test firms).
6. **Imputation + scaling**: median imputation followed by `StandardScaler`, both
   fit on the training split only and applied unchanged to the test split.
7. **Target scaling for the neural models**: the target itself (mean~11, std~7.5)
   is also standardized (fit on the training split) before training the plain NN
   and the hybrid -- see Section 8.3 for why this mattered.

## 7. Feature Engineering

The 25 features used for modeling, grouped by theme:

| Theme | Features |
|---|---|
| Firm profile | `log_workers`, `share_managers`, `share_professionals`, `share_clerical`, `admin_worker_share`, `has_website` |
| AI usage | `share_computer_use`, `uses_genai`, `ai_b2a4_flag`, `ai_breadth_score`, `task_breadth`, `ai_b3a2_flag`, `ai_b3a3_flag` |
| Digitization maturity | `finance_digitization`, `customer_digitization`, `supplychain_digitization` |
| Operating conditions | `idle_time_share`, `workers_wanting_more_hours_share`, `can_absorb_demand_shock`, `reskilling_need_share`, `sales_seasonality_ratio`, `worker_seasonality_ratio` |
| Policy support | `policy_support_score`, `ai_g1b_flag`, `ai_g1f_flag` |

Full definitions are in `src/02_feature_engineering.py`.

## 8. Model Development

### 8.1 Hybrid Wide & Deep architecture

The core modeling contribution is a **Wide & Deep** model (Cheng et al., 2016),
built with the Keras functional API as a single, jointly trained network:

```
                 ┌─────────────────────────┐
   scaled        │   Wide path              │
   input ────────┤   Dense(1, linear)       │───┐
   features      │   (== linear regression) │   │
   (25 cols)     └─────────────────────────┘   │
                 ┌─────────────────────────┐   ▼
                 │   Deep path               │  Add  ──▶ prediction
                 │   Dense(128, relu)+BN+Drop(0.3)│  ▲
                 │   Dense(64,  relu)+BN+Drop(0.2)│  │
                 │   Dense(32,  relu)+BN+Drop(0.1)│  │
                 │   Dense(1)                │──┘
                 └─────────────────────────┘
```

- **Wide component**: a single `Dense(1, linear)` applied directly to the scaled
  inputs -- mathematically equivalent to linear regression, capturing direct
  linear feature-target relationships.
- **Deep component**: 3 hidden layers (128 -> 64 -> 32, ReLU) with
  BatchNormalization and Dropout (0.3 / 0.2 / 0.1) after each layer, capturing
  nonlinear interactions between features.
- The wide and deep outputs are **summed** into one final prediction and trained
  jointly against one Adam optimizer and one MSE loss -- one unified model, not
  two models stacked after the fact.
- Trained with `EarlyStopping` (patience=25, restore best weights) and
  `ReduceLROnPlateau`, using a validation split carved out of the training data.

**Why this architecture**: it satisfies the assignment's requirement to combine
linear regression and deep learning in one model, while keeping the linear
signal directly interpretable through the wide path's weights and letting the
deep path pick up any nonlinear structure the linear path misses.

### 8.2 Baselines compared

- Mean-prediction baseline
- Plain Linear Regression (no regularization)
- Ridge, Lasso (`alpha` selected via cross-validation)
- Random Forest, Gradient Boosting, Extra Trees (each tuned with
  `RandomizedSearchCV`, 3-fold CV, 30 candidate configurations)
- Stacking ensemble (RF + GB + ET base learners, Ridge meta-learner)
- Plain feedforward neural network (the hybrid's deep component alone, trained
  independently, for a controlled comparison)

### 8.3 A convergence bug, caught and fixed before finalizing

An earlier version of this analysis trained the plain NN and hybrid directly on
the *unscaled* target (`ai_f2a2`, mean~11, std~7.5), while the 25 input features
were standardized as usual. Both models scored well below plain Linear
Regression (R^2 ~0.28-0.35 vs Linear's 0.48) -- unusual enough that it warranted
checking rather than accepting as "deep learning's limitation," since an
under-converged network is a far more common explanation for a neural model
losing to plain linear regression than a genuine architectural ceiling.

The diagnosis: loss curves showed clean, stable convergence for both models
(steep drop, plateau, clean early stop -- no divergence), which ruled out a
learning-rate blowup. That pointed instead to the target scale itself: an
unscaled regression target combined with small-magnitude default weight
initialization is a known cause of slow, suboptimal convergence, since the
network has to learn an output bias/scale roughly matching the target's
magnitude before it can start fitting structure. This was confirmed directly:
adding target standardization (fit on the training split, inverse-transformed
before scoring) raised validation R^2 for both models (plain NN 0.46 -> 0.54,
hybrid 0.46 -> 0.49) in a controlled A/B comparison that changed nothing else.

A second issue surfaced once target scaling was in place: single-run test R^2
for these two models varied enormously by random seed (hybrid ranged from 0.23
to 0.52 across 5 seeds on one architecture) -- expected for small networks on a
788-row training set, but it means any single test-set number is close to
meaningless on its own. The fix adopted here: report the **mean test R^2 over 5
independent seeds** (0-4) as the comparison metric for both neural models,
rather than one lucky-or-unlucky run (full per-seed detail in
`reports/deep_model_seed_stability.csv`). A lighter plain-NN architecture
(64->32->16, dropout 0.2/0.1/0.1, selected via a validation-only comparison
against the original 128->64->32) is used for the plain NN; the hybrid keeps its
original architecture, since only the target-scaling fix was being isolated
there.

**Net effect**: target scaling was a real bug, not a red herring. Fixing it
raised the plain NN's mean test R^2 from ~0.32 to ~0.47 -- now within a hair of
Linear Regression's 0.48. The hybrid's mean rose from ~0.28-0.35 to ~0.39. The
qualitative conclusion is unchanged (tree ensembles still win decisively, see
Section 10), but the deep models are no longer being sold short by a fixable
convergence issue -- what's reported below is their genuine, validated ceiling
on this dataset.

## 9. Model Validation

- Fixed 80/20 train-test split, `random_state=42`, identical for every model.
- Hyperparameter tuning used 3-fold cross-validation on the training set only --
  the test set was never touched during tuning.
- Every model was scored with R^2, RMSE, and MAE on the same 198-firm held-out
  test set.

## 10. Results and Discussion

| Model | R^2 | RMSE | MAE |
|---|---|---|---|
| **Extra Trees (tuned)** | **0.621** | **4.04** | **2.87** |
| Stacking (RF+GB+ET -> Ridge) | 0.621 | 4.04 | 2.81 |
| Random Forest (tuned) | 0.617 | 4.06 | 2.80 |
| Gradient Boosting (tuned) | 0.607 | 4.11 | 2.78 |
| Lasso | 0.487 | 4.70 | 3.37 |
| Linear Regression | 0.481 | 4.73 | 3.40 |
| Ridge | 0.480 | 4.73 | 3.40 |
| Plain feedforward NN | 0.465 (±0.054) | 4.79 | 3.40 |
| Hybrid Wide & Deep | 0.395 (±0.054) | 5.10 | 3.57 |
| Mean baseline | -0.003 | 6.57 | 5.41 |

*(Plain NN / Hybrid rows are the mean ± std over 5 seeds, test set, after the
target-scaling fix described in Section 8.3 -- see
`reports/deep_model_seed_stability.csv` for every individual run and
`reports/metrics_deep.csv` for the exact numbers backing this table. All other
rows are deterministic given `random_state=42`.)*

**The tree ensembles win, clearly and consistently.** Extra Trees (tuned) was
selected as the deployed model with test R^2 = 0.621, RMSE = 4.04, MAE = 2.87 --
essentially tied with the Stacking ensemble and Random Forest. After fixing the
target-scaling issue in Section 8.3, the plain NN closed most of its gap with
linear regression (mean R^2 0.465 vs Linear's 0.481 -- within noise of each
other) and the hybrid improved similarly (0.395, up from ~0.28-0.35 pre-fix).
Neither deep model outperforms the tree ensembles.

**Why the hybrid didn't win, honestly**: this is a small tabular dataset (788
training rows, 25 features). Deep learning architectures generally need
substantially more data to learn useful nonlinear representations than tree
ensembles do; tree-based methods handle small-N tabular problems with
mixed-type, non-smooth features very well precisely because they partition the
feature space directly rather than trying to learn a smooth function through
gradient descent. This is a well-documented finding in the ML literature (e.g.
Grinsztajn et al., 2022, on why tree-based models still outperform deep learning
on tabular data), not a flaw in the Wide & Deep implementation -- and, per
Section 8.3, not a convergence artifact either, since that was specifically
checked and fixed first. The hybrid model is included in full per the
assignment's ML+DL requirement, and the comparison above is reported without
adjustment.

Best hyperparameters found (`reports/baseline_best_params.json`):
- Random Forest: 500 trees, max_depth=10, max_features='sqrt'
- Gradient Boosting: 100 trees, max_depth=4, learning_rate=0.05, subsample=0.7
- Extra Trees: 200 trees, max_depth=20, max_features='sqrt'
- Ridge alpha=10, Lasso alpha≈0.032

## 11. Model Interpretation

Permutation importance on the deployed Extra Trees model (test set, mean drop in
R^2 over 15 shuffles per feature; full table in
`reports/permutation_importance.csv`, plot in
`figures/10_permutation_importance.png`):

| Rank | Feature | Importance (ΔR^2) |
|---|---|---|
| 1 | `reskilling_need_share` | 0.128 |
| 2 | `workers_wanting_more_hours_share` | 0.098 |
| 3 | `idle_time_share` | 0.072 |
| 4 | `can_absorb_demand_shock` | 0.068 |
| 5 | `admin_worker_share` | 0.040 |
| 6 | `ai_g1b_flag` (workforce-training policy support) | 0.039 |
| 7 | `ai_b3a3_flag` (AI used for drafting) | 0.031 |
| 8 | `ai_b3a2_flag` (AI used for summarizing) | 0.029 |
| 9 | `customer_digitization` | 0.027 |
| 10 | `ai_b2a4_flag` / `uses_genai` (generative AI use) | 0.022 |

Operating-condition variables (reskilling need, idle capacity, demand-shock
absorption) dominate over pure AI-adoption flags -- suggesting expected AI gains
are driven at least as much by *how much slack and disruption a firm already has
in its operations* as by how much AI it currently uses. See diagnostic plots:
`figures/08_pred_vs_actual.png` (predicted vs. actual) and
`figures/09_residuals.png` (residuals).

## 12. Conclusion

A validated 25-feature pipeline, built from 132 raw survey columns through a
reproducible cleaning and feature-engineering process, supports a regression
model that explains roughly 62% of the variance in firms' self-reported expected
AI productivity gains. Tree ensembles (Extra Trees, in a near-tie with a
Stacking ensemble and Random Forest) outperform both classical linear models and
a purpose-built hybrid Wide & Deep neural network on this dataset -- a
conclusion checked, not assumed: an initial version of the deep models
under-converged due to an unscaled training target, and was corrected and
re-validated (Section 8.3) before this result was finalized. The winning model
is deployed as an interactive Streamlit application that estimates a firm's
expected AI productivity gain from a short wizard of inputs.

## 13. Limitations

- The target is a **subjective, self-reported expectation**, not a measured
  outcome -- it reflects respondent beliefs and optimism about AI, not realized
  productivity.
- R^2 of 0.62 is a strong, honest result for this kind of noisy survey target;
  much higher would likely indicate leakage rather than genuine skill.
- The modeling sample (986 of 1,355 firms) and training set (788 firms) are
  small -- deep-learning results in particular carry meaningful run-to-run
  variance at this scale.
- Cross-sectional data: reported associations are not causal claims.

## 14. Future Scope

- A longitudinal follow-up survey, comparing expectations to realized outcomes,
  would let the target be validated or recalibrated.
- A larger sample would materially help the deep-learning components, which are
  more data-hungry than the tree ensembles that currently win.
- Sector/industry fixed effects and interaction terms (e.g., AI breadth x firm
  size) may capture structure the current linear + tree feature set misses.

## 15. References

- Cheng, H.-T., Koc, L., Harmsen, J., et al. (2016). *Wide & Deep Learning for
  Recommender Systems.* arXiv:1606.07792.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models
  still outperform deep learning on tabular data?* NeurIPS Datasets and
  Benchmarks Track.

## Appendix

- Full engineered feature table (59 columns): `data/processed/features_full.csv`
- EDA figures: `figures/01`-`05`
- Deep-learning training curves: `figures/06_plain_nn_loss.png`,
  `figures/07_hybrid_loss.png`
- Diagnostic plots: `figures/08_pred_vs_actual.png`, `figures/09_residuals.png`,
  `figures/10_permutation_importance.png`
- Full model comparison table: `reports/model_comparison.csv`
- Best hyperparameters: `reports/baseline_best_params.json`
- Permutation importance (all features): `reports/permutation_importance.csv`
- EDA summary statistics: `reports/eda_summary_stats.csv`
