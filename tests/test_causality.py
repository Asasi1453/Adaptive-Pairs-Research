"""Regression checks using synthetic prices; no market data or network needed."""
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import clean_bad_prints, load_pair
from engine import _size_factors, run_backtest, Costs
from signals import kalman_signal, wavelet_signal


class CausalityTests(unittest.TestCase):
    def test_future_beta_cannot_change_past_position_sizes(self):
        rng = np.random.default_rng(1453)
        past = np.cumsum(rng.normal(0, 0.01, 150)) + 0.6
        future = np.cumsum(rng.normal(0, 1, 150)) + past[-1]
        short = _size_factors(past, 40, 0.1, 1.0)
        long = _size_factors(np.r_[past, future], 40, 0.1, 1.0)
        np.testing.assert_allclose(short, long[:len(past)], rtol=0, atol=1e-12)

    def test_future_beta_cannot_change_an_already_closed_trade(self):
        rng = np.random.default_rng(12)
        beta = np.r_[0.6 + np.cumsum(rng.normal(0, 0.01, 150)),
                     0.6 + np.cumsum(rng.normal(0, 0.8, 150))]
        prices = pd.DataFrame({"price_v": np.linspace(100, 110, 300),
                               "price_ma": np.full(300, 100.0)},
                              index=pd.date_range("2025-01-02", periods=300, freq="min", tz="UTC"))
        z = np.zeros(300); z[80:85] = -3.0
        small, _ = run_backtest(prices.iloc[:150], z[:150], beta[:150], 2, 0.5)
        full, _ = run_backtest(prices, z, beta, 2, 0.5)
        pd.testing.assert_frame_equal(small, full)

    def test_default_loader_preserves_a_jump_seen_before_a_reversal(self):
        times = pd.date_range("2025-01-02 15:00", periods=3, freq="min", tz="UTC")
        records = [{"timestamp": t, "ticker": ticker, "close": value}
                   for t, v in zip(times, [100, 120, 100])
                   for ticker, value in [("V", v), ("MA", 100)]]
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "pair.parquet"
            pd.DataFrame(records[:4]).to_parquet(p)
            prefix = load_pair(str(p), resample="", regular_hours_only=False)
            pd.DataFrame(records).to_parquet(p)
            full = load_pair(str(p), resample="", regular_hours_only=False)
        pd.testing.assert_frame_equal(prefix, full.iloc[:2])

    def test_retrospective_cleaner_warns_and_remains_opt_in(self):
        frame = pd.DataFrame({"price_v": [100, 120, 100], "price_ma": [100, 100, 100]},
                             index=pd.date_range("2025-01-02", periods=3, tz="UTC"))
        with self.assertWarnsRegex(UserWarning, "future"):
            result = clean_bad_prints(frame, verbose=False)
        self.assertEqual(list(result.index), [frame.index[0], frame.index[2]])

    def test_sizes_are_finite_bounded_with_flat_beta(self):
        sizes = _size_factors(np.full(80, 0.6), 40, 0.1, 1.0)
        self.assertTrue(np.all(np.isfinite(sizes)))
        self.assertTrue(np.all((sizes >= 0.1) & (sizes <= 1.0)))

    def test_signal_prefix_invariance(self):
        rng = np.random.default_rng(4)
        ma = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 180)))
        v = 200 * np.exp(np.cumsum(rng.normal(0, 0.001, 180)))
        for signal, kwargs in [(kalman_signal, {}), (wavelet_signal, {"window": 64})]:
            with self.subTest(method=signal.__name__):
                short = signal(v[:130], ma[:130], **kwargs)
                full = signal(v, ma, **kwargs)
                for left, right in zip(short, full):
                    np.testing.assert_allclose(left, right[:130], rtol=0, atol=1e-10)

    def test_next_bar_execution_and_four_commissions(self):
        frame = pd.DataFrame({"price_v": [100., 100., 110., 110.], "price_ma": [100.] * 4},
                             index=pd.date_range("2025-01-02", periods=4, freq="min", tz="UTC"))
        trades, equity = run_backtest(frame, np.array([-3., -3., 0., 0.]),
                                     np.ones(4), 2, 0.5,
                                     Costs(initial_capital=10000, slippage_bps=0,
                                           spread_cents=0, commission_per_fill=1.5))
        self.assertEqual(trades.iloc[0].entry_time, frame.index[1])
        self.assertEqual(trades.iloc[0].exit_time, frame.index[3])
        self.assertAlmostEqual(trades.iloc[0].net_pnl, 50 * 10 - 6)
        self.assertAlmostEqual(equity.iloc[-1], 10494)


if __name__ == "__main__":
    unittest.main()
