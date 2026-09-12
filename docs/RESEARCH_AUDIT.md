# Methodology review — 13 September 2026

Base revision: `3b970dd443741436cc8fc17b1b5a6c5e291742fa`.

## Corrected in this change

1. `engine._size_factors` used `np.nanmedian(vol)` on the complete period. Replaced with an expanding median shifted by one bar. Warmup retains the existing fallback sizing policy; it is not a new estimate of optimal risk.
2. `data.load_pair` enabled reversal-based cleaning by default. That cleaner uses the next return, which changes past input rows when future rows arrive. Default is now `clean=False`; explicit retrospective cleaning emits a warning. A price reversal alone cannot establish a bad print.

Regression verification: seven unittest cases; the pre-change code failed four checks, including a closed-trade invariance counterexample. The revised code passed all seven. Signal-prefix checks concern the estimators given fixed inputs, not the entire pipeline.

## Still open before a new performance claim

- **Execution information set:** `pos` is delayed, while the engine reads `beta[entry_i]` and `size_f[entry_i]`. Freeze order parameters consistently at the intended decision time and test that execution-bar information cannot alter a precommitted order.
- **Coefficient units:** a log-return beta is a notional exposure ratio, while the engine treats beta as a share-count ratio. For that model, the corresponding magnitude is `q_MA / q_V = beta * P_V / P_MA`; a level-regression beta has different units. Define the conventions before changing performance calculations.
- **Equity accounting:** current equity includes only initial capital and realized balances at exits. It cannot measure intra-trade drawdown. Add bar-level mark-to-market holdings and reconcile final equity to the trade ledger.
- **Entry point consistency:** `run.py` does not pass the session-gap mask or the beta-valid entry filter used by `wavelet_1min.py`. Use a common signal/evaluation configuration.
- **Timestamp semantics:** the [upstream dataset card](https://huggingface.co/datasets/mito0o852/OHLCV-1m) describes timestamps as the start of the minute. Close values therefore become available later; resample labels must represent actual data availability, including partial bars and the temporal split boundary.
- **Validation:** repeatedly inspecting and revising a full-period run can overfit methodology even with fixed entry thresholds. Preserve a research log and label the existing period retrospective.
- **Data provenance:** document dataset revision, corporate-action treatment, duplicate/missing-bar policy, exchange sessions, file checksums and conversion steps. A deterministic synthetic check is not market-data reproduction.
- **Economic model:** borrowing, dividends on short holdings, financing, liquidity and impact need explicit treatment or explicit limitations.

## Interpreting existing outputs

Previously stored CSV, JSON, plots and the legacy README predate the corrections. They remain historical experiment records. Do not quote them as corrected out-of-sample returns or as verified maximum drawdown. `gross_pnl` in the current engine already includes spread/slippage; it is pre-commission rather than before all costs.
