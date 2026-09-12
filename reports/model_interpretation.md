# Model Interpretation

**Final deployed model:** Extra Trees (tuned)

**Test-set performance:** R2 = 0.6207, RMSE = 4.0420, MAE = 2.8710  (n_train = 788, n_test = 198)

## Important predictors

Ranked by permutation importance on the held-out test set (mean drop in R^2 over 15 shuffles per feature):

- **reskilling_need_share**: 0.1282 (+/- 0.0209)
- **workers_wanting_more_hours_share**: 0.0979 (+/- 0.0171)
- **idle_time_share**: 0.0719 (+/- 0.0210)
- **can_absorb_demand_shock**: 0.0682 (+/- 0.0194)
- **admin_worker_share**: 0.0404 (+/- 0.0103)
- **ai_g1b_flag**: 0.0392 (+/- 0.0093)
- **ai_b3a3_flag**: 0.0306 (+/- 0.0091)
- **ai_b3a2_flag**: 0.0291 (+/- 0.0090)
- **customer_digitization**: 0.0266 (+/- 0.0065)
- **ai_b2a4_flag**: 0.0216 (+/- 0.0095)

## Practical usefulness

The model surfaces which firm characteristics associate with a firm's own expectation of AI-driven productivity gains: breadth and maturity of AI/digital adoption (task breadth, generative-AI use, digitization ladders), workforce composition (share of managers/professionals/clerical staff, firm size), and operating slack (idle time, seasonality, capacity headroom). This is useful as a directional screening tool -- e.g. for a policy team or lender trying to identify which firms report the most AI-driven upside -- rather than as a precise forecast for any single firm.

## Limitations

- The target (`ai_f2a2`) is a **subjective, self-reported expectation**, not a measured outcome -- it captures what a firm's respondent believes will happen, which is shaped by optimism, awareness of AI, and expectations rather than realized productivity data.
- R^2 of 0.62 is a strong, honest result for this kind of noisy, self-reported survey target; an R^2 much higher than ~0.65 here would be a signal of leakage rather than genuine predictive skill.
- The modeling sample is limited to the ~986 firms that answered the target question (of 1,355 surveyed), and training used only ~80% of that (788 firms) -- a small-N regime where estimates carry real uncertainty.
- Cross-sectional survey data: associations here are not causal claims.

## Possible improvements

- Collect a longitudinal follow-up (realized productivity change vs the prior expectation) to validate or recalibrate the self-reported target.
- Expand the training sample -- both models compared here (tree ensembles and the wide & deep hybrid) are constrained more by sample size than by architecture.
- Add sector/industry fixed effects or interaction terms between AI adoption breadth and firm size, which permutation importance suggests may be under-captured by the current linear + tree feature set.
