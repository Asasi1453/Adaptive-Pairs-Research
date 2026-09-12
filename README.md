# Adaptive Pairs Research

**V/MA pairs trading with Kalman and wavelet estimators.**

An exploratory comparison of adaptive hedge estimates for Visa and Mastercard, with a focus on how data handling, execution assumptions and transaction costs affect conclusions.

**Status:** methodology under revision. Historical return tables predate the September 2026 review and should not be treated as validated performance. Two future-data dependencies have been corrected; additional modeling issues remain documented below.

## Research question

Does wavelet-denoised return estimation offer a useful advantage over simpler hedge estimates once evaluation is chronological and trading assumptions are explicit?

The current implementation contains a Kalman price-level model and a wavelet-denoised return model. They are different models with different coefficient units, so their conversion into tradable positions needs further work before a fair performance comparison.

## Start with the evidence

- [Research note and next experiment](docs/RESEARCH_NOTE.md)
- [Methodology review and remaining issues](docs/RESEARCH_AUDIT.md)
- [Regression checks](tests/test_causality.py)
- [Original Turkish write-up — legacy, not current validation](README.legacy.md)

## Run the small reproducibility check

Run from the repository root. No downloaded market data are needed for these synthetic regression checks.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ci.txt
python -m unittest discover -s tests -v
```

The checks cover historical position-size invariance when future observations are appended, already-closed trade invariance, default loader behavior around a jump/reversal, warnings on retrospective cleaning, finite sizing, signal-prefix invariance, next-bar execution and four-fill commissions. They do not certify the entire backtest as causal or economically correct.

## Market-data experiments

Install `requirements.txt` for the plotting and research scripts. The original data loader expects a Parquet file with `timestamp`, `ticker`, and `close` columns for V and MA. A processed V/MA snapshot is tracked at `data/processed/pair_V_MA.parquet` (1,038,776 rows before filtering and alignment). See [the data manifest](docs/DATA_MANIFEST.json) for its SHA-256 and source. A complete upstream ingestion recipe and source revision were not recorded. The existing `.gitignore` rule does not untrack this already committed file.

From the repository root, existing exploratory entry points are:

```sh
python src/run.py --data data/processed/pair_V_MA.parquet --out-dir results
python src/wavelet_1min.py --data data/processed/pair_V_MA.parquet --out-dir results_1min
```

These are legacy research entry points, not a completed confirmatory protocol. The first selects thresholds on a chronological training portion. The one-minute script evaluates fixed thresholds over the full period; fixed thresholds alone do not make it out-of-sample.

The default loader now preserves price jumps. Explicit `clean=True` performs retrospective diagnostics using the next observation and emits a warning; it is unsuitable as evidence of causal trading performance.

## Implementation map

| File | Responsibility |
|---|---|
| `src/data.py` | Alignment, session handling and optional retrospective diagnostics |
| `src/signals.py` | Kalman and wavelet estimators |
| `src/engine.py` | Positions, sizing, fills and trade ledger |
| `src/metrics.py` | Summary statistics from the current ledger |
| `src/run.py` | Existing five-minute threshold search and temporal split |
| `src/wavelet_1min.py` | Existing full-period exploratory analysis |

Before reporting updated performance, resolve hedge-ratio units, signal/execution parameter timing, bar-level equity accounting and consistent session/beta filters across entry points. Then record data provenance and rerun a frozen evaluation protocol. No claim of a profitable trading strategy is made.
