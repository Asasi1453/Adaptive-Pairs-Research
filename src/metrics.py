"""Performans metrikleri."""

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, equity: pd.Series, initial_capital: float,
              label: str = "") -> dict:
    """Tek bir kosunun ozeti."""
    n = len(trades)
    if n == 0:
        return dict(label=label, total_trades=0, total_return_pct=0.0,
                    final_capital=initial_capital)

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    final_capital = float(trades["capital_after"].iloc[-1])
    gross_win = float(wins["net_pnl"].sum())
    gross_loss = float(abs(losses["net_pnl"].sum()))

    run_max = equity.cummax()
    dd_pct = ((equity - run_max) / run_max * 100).min()

    # trade getirileri uzerinden Sharpe (yillikllastirilmamis, trade-bazli)
    r = trades["return_pct"].to_numpy()
    sharpe_trade = float(r.mean() / r.std()) if len(r) > 1 and r.std() > 0 else 0.0

    return {
        "label": label,
        "total_trades": n,
        "win_rate_pct": len(wins) / n * 100,
        "avg_net_pnl": float(trades["net_pnl"].mean()),
        "avg_return_pct": float(trades["return_pct"].mean()),
        "avg_win": float(wins["net_pnl"].mean()) if len(wins) else np.nan,
        "avg_loss": float(losses["net_pnl"].mean()) if len(losses) else np.nan,
        "profit_factor": gross_win / gross_loss if gross_loss else np.inf,
        "sharpe_per_trade": sharpe_trade,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "total_commission": float(trades["commission"].sum()),
        "total_slippage_spread": float(trades["slippage_spread_cost"].sum()),
        "total_gross_pnl": float(trades["gross_pnl"].sum()),
        "total_net_pnl": float(trades["net_pnl"].sum()),
        "final_capital": final_capital,
        "total_return_pct": (final_capital / initial_capital - 1) * 100,
        "max_drawdown_pct": float(dd_pct),
    }


def yearly_pnl(trades: pd.DataFrame) -> pd.Series:
    if len(trades) == 0:
        return pd.Series(dtype=float)
    return trades.groupby(trades["exit_time"].dt.year)["net_pnl"].sum()


def format_summary(s: dict) -> str:
    if s.get("total_trades", 0) == 0:
        return f"  {s.get('label','')}: trade yok"
    return (
        f"  trade={s['total_trades']}  win={s['win_rate_pct']:.1f}%  "
        f"getiri={s['total_return_pct']:+.2f}%  maxDD={s['max_drawdown_pct']:.2f}%\n"
        f"  brut PnL=${s['total_gross_pnl']:,.0f}  komisyon=${s['total_commission']:,.0f}  "
        f"slipaj+spread=${s['total_slippage_spread']:,.0f}  net=${s['total_net_pnl']:,.0f}\n"
        f"  trade basi: ${s['avg_net_pnl']:,.2f} ({s['avg_return_pct']:+.3f}%)  "
        f"PF={s['profit_factor']:.2f}  ort.sure={s['avg_bars_held']:.0f} bar"
    )
