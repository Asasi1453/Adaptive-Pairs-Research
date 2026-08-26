"""
WAVELET stratejisi - 1 DAKIKALIK bar analizi.

Sabit parametreler: entry_z = 2.0 sigma, exit_z = 0.5 sigma.
Parametreler onceden sabitlendigi (optimize edilmedigi) icin train/test
ayrimina gerek yok - hicbir sey veriye fit edilmiyor, dolayisiyla TUM
donem (2021-01 .. 2026-03) uzerinde calistirilir.

Pencere: 1950 bar. 5-dakikalik kosudaki 390 barlik pencere ile AYNI
ekonomik geriye bakisi (1 islem haftasi) korur: 390 x 5dk = 1950 x 1dk.

Uretilenler (hepsi --out-dir klasorune):
  stats.json / stats.txt      - tum ozet metrikler
  trades.csv                  - trade listesi
  pnl_per_trade.png           - trade basina net PnL (bar chart)
  equity_curve_pct.png        - yuzdesel sermaye egrisi + drawdown
  trade_duration.png          - trade basina sure (bar chart) + dagilim
  beta_over_time.png          - beta'nin zamana bagli grafigi
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from data import load_pair, session_gap_mask
from signals import wavelet_signal
from engine import run_backtest, Costs
from metrics import summarize

# ---- gorsel kimlik ----
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, GOOD, BAD = "#2a78d6", "#0ca30c", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.5, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def drawdown_pct(equity: pd.Series) -> pd.Series:
    return (equity - equity.cummax()) / equity.cummax() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed/pair_V_MA.parquet")
    ap.add_argument("--out-dir", default="../results_1min")
    ap.add_argument("--window", type=int, default=1950)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--slippage-bps", type=float, default=2.0, help="grafiklerde kullanilan senaryo")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- veri & sinyal ----------------
    df = load_pair(args.data, resample="1min")
    pv, pm = df["price_v"].to_numpy(), df["price_ma"].to_numpy()
    print(f"1-dakikalik: {len(df):,} bar  {df.index.min()} -> {df.index.max()}")
    print(f"pencere={args.window} bar (~1 islem haftasi)  entry={args.entry_z}s  exit={args.exit_z}s")
    print("wavelet sinyali hesaplaniyor...", flush=True)

    beta, z = wavelet_signal(pv, pm, window=args.window,
                             gap_mask=session_gap_mask(df.index))
    active = beta != 0

    # hedge orani saglikli araligin disindaysa yeni giris acma (mevcut
    # pozisyon tutulur - zorla kapatmak gereksiz round-trip maliyeti uretir)
    lo, hi = CFG.beta_valid
    entry_mask = (beta >= lo) & (beta <= hi)
    print(f"  giris izni: {100*entry_mask[active].mean():.1f}% bar "
          f"(beta {lo}-{hi} disindaki barlarda yeni giris yok)")
    print(f"  beta: medyan={np.median(beta[active]):.3f}  negatif={100*(beta<0).mean():.2f}%")
    print(f"  z   : |z|>2={100*(np.abs(z)>2).mean():.2f}%  std={z.std():.2f}")

    # ---------------- backtest (3 slipaj senaryosu) ----------------
    runs = {}
    for slip in CFG.report_slippage_bps:
        c = Costs(initial_capital=CFG.initial_capital, slippage_bps=slip,
                  spread_cents=CFG.spread_cents, commission_per_fill=CFG.commission_per_fill)
        trades, equity = run_backtest(df, z, beta, args.entry_z, args.exit_z, c,
                                      exec_lag=1, entry_mask=entry_mask)
        runs[slip] = (trades, equity, summarize(trades, equity, CFG.initial_capital, f"slip{slip:g}"))

    trades, equity, S = runs[args.slippage_bps]
    trades.to_csv(f"{args.out_dir}/trades.csv", index=False)

    # ---------------- sure metrikleri ----------------
    # 1-dakikalik bar + sadece normal seans => bars_held = piyasa (market) dakikasi
    market_min = trades["bars_held"].to_numpy().astype(float)
    wall_min = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds().to_numpy() / 60

    dur = {
        "avg_market_minutes": float(market_min.mean()),
        "median_market_minutes": float(np.median(market_min)),
        "min_market_minutes": float(market_min.min()),
        "max_market_minutes": float(market_min.max()),
        "avg_market_hours": float(market_min.mean() / 60),
        "avg_market_trading_days": float(market_min.mean() / 390),
        "avg_wall_clock_minutes": float(wall_min.mean()),
        "avg_wall_clock_days": float(wall_min.mean() / 60 / 24),
    }

    dd = drawdown_pct(equity)
    eq_pct = (equity / CFG.initial_capital - 1) * 100

    stats = {
        "setup": {
            "bar": "1min", "window_bars": args.window,
            "entry_z": args.entry_z, "exit_z": args.exit_z,
            "period": [str(df.index.min()), str(df.index.max())],
            "bars": len(df), "initial_capital": CFG.initial_capital,
            "commission_per_fill": CFG.commission_per_fill,
            "spread_cents": CFG.spread_cents, "exec_lag_bars": 1,
            "note": "parametreler sabit (optimize edilmedi) -> tum donem kullanildi",
        },
        "by_slippage": {f"{s:g}/10000": {
            "total_trades": r[2]["total_trades"],
            "win_rate_pct": round(r[2]["win_rate_pct"], 2),
            "total_return_pct": round(r[2]["total_return_pct"], 2),
            "max_drawdown_pct": round(r[2]["max_drawdown_pct"], 2),
            "profit_factor": round(r[2]["profit_factor"], 3),
            "net_pnl": round(r[2]["total_net_pnl"], 2),
            "gross_pnl": round(r[2]["total_gross_pnl"], 2),
            "commission": round(r[2]["total_commission"], 2),
            "slippage_spread": round(r[2]["total_slippage_spread"], 2),
        } for s, r in runs.items()},
        "duration": {k: round(v, 2) for k, v in dur.items()},
        "beta": {
            "median": round(float(np.median(beta[active])), 4),
            "mean": round(float(beta[active].mean()), 4),
            "std": round(float(beta[active].std()), 4),
            "min": round(float(beta[active].min()), 4),
            "max": round(float(beta[active].max()), 4),
            "pct_negative": round(float(100 * (beta < 0).mean()), 4),
        },
    }
    with open(f"{args.out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---------------- konsol raporu ----------------
    lines = []
    lines.append("=" * 68)
    lines.append(f"WAVELET · 1-DAKIKALIK · entry={args.entry_z}σ / exit={args.exit_z}σ")
    lines.append("=" * 68)
    lines.append(f"Donem      : {df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d}  ({len(df):,} bar)")
    lines.append(f"Sermaye    : ${CFG.initial_capital:,.0f} | komisyon ${CFG.commission_per_fill}/fill (${CFG.commission_per_fill*4:.0f}/trade)")
    lines.append(f"Pencere    : {args.window} bar (~1 islem haftasi) | emir gecikmesi 1 bar")
    lines.append("")
    lines.append(f"{'slipaj':>10} {'trade':>7} {'win%':>7} {'getiri%':>9} {'maxDD%':>8} {'PF':>6} {'net $':>11}")
    for s, (t, e, m) in runs.items():
        lines.append(f"{s:>7g}/1e4 {m['total_trades']:>7} {m['win_rate_pct']:>7.1f} "
                     f"{m['total_return_pct']:>+9.2f} {m['max_drawdown_pct']:>8.2f} "
                     f"{m['profit_factor']:>6.2f} {m['total_net_pnl']:>11,.0f}")
    lines.append("")
    lines.append(f"--- TRADE SURESI (slipaj {args.slippage_bps:g}/10000, {len(trades)} trade) ---")
    lines.append(f"  Ortalama MARKET TIME : {dur['avg_market_minutes']:,.1f} dk "
                 f"({dur['avg_market_hours']:.2f} saat = {dur['avg_market_trading_days']:.2f} islem gunu)")
    lines.append(f"  Medyan market time   : {dur['median_market_minutes']:,.1f} dk")
    lines.append(f"  Min / Max            : {dur['min_market_minutes']:,.0f} dk / {dur['max_market_minutes']:,.0f} dk")
    lines.append(f"  Ortalama duvar saati : {dur['avg_wall_clock_minutes']:,.1f} dk ({dur['avg_wall_clock_days']:.2f} takvim gunu)")
    lines.append("")
    lines.append(f"--- BETA ---")
    b = stats["beta"]
    lines.append(f"  medyan={b['median']}  ort={b['mean']}  std={b['std']}  "
                 f"min={b['min']}  max={b['max']}  negatif={b['pct_negative']}%")
    report = "\n".join(lines)
    print("\n" + report)
    with open(f"{args.out_dir}/stats.txt", "w") as f:
        f.write(report + "\n")

    # ================= GRAFIKLER =================
    tag = f"slipaj {args.slippage_bps:g}/10000"

    # 1) trade basina net PnL
    fig, ax = plt.subplots(figsize=(15, 5.5))
    pnl = trades["net_pnl"].to_numpy()
    ax.bar(range(len(pnl)), pnl, color=[GOOD if v > 0 else BAD for v in pnl], width=1.0, linewidth=0)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(f"Trade başına net PnL — {len(pnl)} trade · {tag}\n"
                 f"kazanan {int((pnl>0).sum())} · kaybeden {int((pnl<=0).sum())} · "
                 f"ortalama \\${pnl.mean():,.2f} · toplam \\${pnl.sum():,.0f}")
    ax.set_xlabel("trade sırası"); ax.set_ylabel("net PnL ($)")
    ax.set_xlim(-1, len(pnl))
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/pnl_per_trade.png", dpi=130); plt.close(fig)

    # 2) yuzdesel sermaye egrisi + drawdown
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    for s, (t, e, m) in runs.items():
        pct = (e / CFG.initial_capital - 1) * 100
        lw, alpha = (2.0, 1.0) if s == args.slippage_bps else (1.2, 0.65)
        a1.plot(pct.index, pct.values, lw=lw, alpha=alpha,
                label=f"slipaj {s:g}/10000   {m['total_return_pct']:+.2f}%  (maxDD {m['max_drawdown_pct']:.2f}%)")
    a1.axhline(0, color=MUTED, ls=":", lw=1)
    a1.set_title(f"Sermaye eğrisi (yüzdesel) — Wavelet · 1-dakikalık · entry={args.entry_z}σ / exit={args.exit_z}σ")
    a1.set_ylabel("getiri (%)")
    a1.yaxis.set_major_formatter(lambda x, p: f"{x:+.0f}%")
    a1.legend(frameon=False, fontsize=9, loc="upper left")

    a2.fill_between(dd.index, dd.values, 0, color=BAD, alpha=0.25)
    a2.plot(dd.index, dd.values, color=BAD, lw=1.0)
    a2.axhline(S["max_drawdown_pct"], color=BAD, ls="--", lw=1,
               label=f"max drawdown {S['max_drawdown_pct']:.2f}%")
    a2.set_title(f"Drawdown ({tag})", fontsize=10)
    a2.set_ylabel("drawdown (%)")
    a2.yaxis.set_major_formatter(lambda x, p: f"{x:.0f}%")
    a2.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/equity_curve_pct.png", dpi=130); plt.close(fig)

    # 3) trade suresi
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [2.4, 1]})
    a1.bar(range(len(market_min)), market_min, color=BLUE, width=1.0, linewidth=0)
    a1.axhline(dur["avg_market_minutes"], color=BAD, ls="--", lw=1.2,
               label=f"ortalama {dur['avg_market_minutes']:,.0f} dk")
    a1.axhline(390, color=MUTED, ls=":", lw=1, label="1 işlem günü (390 dk)")
    a1.set_title(f"Trade başına market time — {tag}")
    a1.set_xlabel("trade sırası"); a1.set_ylabel("market time (dakika)")
    a1.set_xlim(-1, len(market_min)); a1.legend(frameon=False, fontsize=9)

    a2.hist(market_min, bins=50, color=BLUE, edgecolor="white", linewidth=0.4)
    a2.axvline(dur["avg_market_minutes"], color=BAD, ls="--", lw=1.2, label="ortalama")
    a2.axvline(dur["median_market_minutes"], color=GOOD, ls="--", lw=1.2, label="medyan")
    a2.set_title("Süre dağılımı"); a2.set_xlabel("market time (dakika)"); a2.set_ylabel("trade sayısı")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/trade_duration.png", dpi=130); plt.close(fig)

    # 4) beta zaman serisi
    fig, ax = plt.subplots(figsize=(15, 5))
    bs = pd.Series(np.where(active, beta, np.nan), index=df.index)
    ax.plot(bs.index, bs.values, color=BLUE, lw=0.35, alpha=0.55, label="beta (1-dk)")
    ax.plot(bs.index, bs.rolling(1950, min_periods=100).mean().values,
            color="#184f95", lw=1.8, label="beta — 1 haftalık ortalama")
    ax.axhline(stats["beta"]["median"], color=BAD, ls="--", lw=1.2,
               label=f"medyan {stats['beta']['median']:.3f}")
    ax.axhline(0, color=INK, lw=0.8, ls=":")
    ax.set_title("Hedge ratio (beta) zaman içinde — V ≈ β × MA, log getiriler üzerinden")
    ax.set_ylabel("beta"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/beta_over_time.png", dpi=130); plt.close(fig)

    print(f"\nKaydedildi -> {args.out_dir}/")
    for f_ in ["stats.json", "stats.txt", "trades.csv", "pnl_per_trade.png",
               "equity_curve_pct.png", "trade_duration.png", "beta_over_time.png"]:
        print(f"  {f_}")


if __name__ == "__main__":
    main()
