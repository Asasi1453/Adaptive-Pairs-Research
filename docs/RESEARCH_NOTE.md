# Adaptive hedge estimation under realistic evaluation

Bilgehan Erdemir · Research discussion draft · 13 September 2026

## Question and motivation

Does denoising improve hedge estimation on financial time series, and does any statistical benefit survive explicit execution costs? Visa/Mastercard provides a concrete setting to compare sequential estimators. Economic similarity is a motivation for choosing the pair, not proof of cointegration or a tradable mean-reverting relationship.

## Existing study

The implementation compares a Kalman price-level model, `V_t = alpha_t + beta_t MA_t + error_t`, with a wavelet-denoised log-return regression. Trading signals are constructed from standardized deviations. The framework includes delayed execution and costs on both legs.

A stored, pre-review experiment reports 485,613 aligned one-minute observations spanning January 2021–March 2026 and 1,113 trades. Its full-period returns at 1, 2 and 3 basis points of assumed slippage were +9.14%, -7.57% and -21.78%, respectively. These are archived outputs, not results reproduced in this review, and not independent out-of-sample estimates. The actual slippage achievable in trading has not been established.

## What the review changed

Two future-data dependencies were demonstrated with small synthetic counterexamples. First, full-sample beta-volatility normalization allowed future observations to alter previously closed trade sizes. The reference now uses the median of volatility observations available strictly before the current bar. Second, automatic reversal-based cleaning deleted an observation after inspecting its successor. That cleaning is now disabled by default and explicitly identified as retrospective diagnostics.

Seven regression tests pass after these changes. Remaining concerns include coefficient units in position construction, the timing of hedge parameters relative to execution, and equity measured only at trade exits. Accordingly, the archived performance should be re-evaluated after the remaining issues are resolved.

## Proposed internship experiment

**Hypothesis.** Denoising can improve hedge-estimate stability, but its economic value is uncertain after costs and chronological evaluation.

**Baselines.** Compare fixed hedge, rolling raw-return OLS and wavelet-denoised return regression using identical data and position accounting. Treat the Kalman level model as a separately specified benchmark with explicit conversion from coefficient units to holdings.

**Protocol.** Freeze preprocessing, forecast/decision times, coefficient-to-position mapping, parameter grids, and primary outcomes before evaluation. Use chronological walk-forward folds, choosing settings on each training window and evaluating only on the subsequent window. Historical periods already inspected remain retrospective evidence; reserve a genuinely untouched future period for confirmation. Never use random row splits for this task.

**Outcomes.** Report hedge error, turnover, net P&L, bar-level mark-to-market drawdown, and sensitivity to costs and execution delays. Use paired comparisons on aligned observations; assess uncertainty with a dependence-aware block bootstrap and sensitivity to block length. Include a zero-position comparator. Treat significance claims as exploratory when multiple specifications have been examined.

**Deliverables.** A versioned data manifest, one-command experiment runner, predeclared configuration, machine-readable results, and a short report including negative outcomes and failed hypotheses.

## Why this is a research contribution

The intended contribution is a careful comparison of estimation and evaluation choices. A stable estimate or an attractive backtest is insufficient by itself. A useful result would show which claimed improvements survive data, accounting, and validation checks—and which do not.
