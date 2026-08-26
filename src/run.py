"""
Tek giris noktasi. Protokol:

  1) Sinyaller TAM veri uzerinde causal olarak hesaplanir (t anindaki deger
     yalnizca gecmisi kullanir).
  2) TRAIN (2021-01 .. 2023-06) uzerinde entry_z x exit_z grid search.
     Secim kriteri: en yuksek toplam getiri, en az CFG.min_trades trade.
  3) Secilen TEK parametre TEST (2023-07 .. 2026-03) doneminde, 3 slipaj
     senaryosuyla calistirilir. Raporlanan "sonuc" budur.

Grid'in kenarina yapisan bir optimum (entry_z veya exit_z sinirda) UYARI
olarak isaretlenir - bu genellikle gercek bir optimum degil, "daha az islem
yap" egiliminin sinira dayanmasidir.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from config import CFG
from data import load_pair, train_test_masks
from signals import SIGNALS
from engine import run_backtest, Costs
from metrics import summarize, yearly_pnl, format_summary


def grid_search(df, z, beta, costs, entry_grid, exit_grid, min_trades):
    rows = []
    for e, x in itertools.product(entry_grid, exit_grid):
        if x >= e:
            continue
        trades, equity = run_backtest(df, z, beta, e, x, costs)
        s = summarize(trades, equity, costs.initial_capital)
        s["entry_z"], s["exit_z"] = e, x
        rows.append(s)
    res = pd.DataFrame(rows)
    return res, res[res["total_trades"] >= min_trades]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=CFG.data_path)
    ap.add_argument("--out-dir", default=CFG.out_dir)
    ap.add_argument("--methods", nargs="+", default=list(SIGNALS.keys()))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_pair(args.data)
    price_v = df["price_v"].to_numpy()
    price_ma = df["price_ma"].to_numpy()
    train_mask, test_mask = train_test_masks(df.index)

    print(f"Veri : {len(df)} bar  {df.index.min()} -> {df.index.max()}")
    print(f"TRAIN: {train_mask.sum():>6} bar  {df.index[train_mask].min()} -> {df.index[train_mask].max()}")
    print(f"TEST : {test_mask.sum():>6} bar  {df.index[test_mask].min()} -> {df.index[test_mask].max()}")
    print(f"Sermaye ${CFG.initial_capital:,.0f} | komisyon ${CFG.commission_per_fill}/fill "
          f"(= ${CFG.commission_per_fill*4:.0f}/trade) | spread {CFG.spread_cents}c")

    select_costs = Costs(initial_capital=CFG.initial_capital,
                         slippage_bps=CFG.select_slippage_bps,
                         spread_cents=CFG.spread_cents,
                         commission_per_fill=CFG.commission_per_fill)

    all_signals, chosen, oos_rows, yearly = {}, {}, [], {}

    for method in args.methods:
        print(f"\n{'='*70}\n{method.upper()}\n{'='*70}")
        print("sinyal hesaplaniyor...", flush=True)
        beta, z = SIGNALS[method](price_v, price_ma)
        all_signals[method] = (beta, z)
        print(f"  beta: medyan={np.median(beta[beta!=0]):.3f}  negatif={100*(beta<0).mean():.2f}%")
        print(f"  z   : |z|>2 orani={100*(np.abs(z)>2).mean():.2f}%  std={z.std():.2f}")

        # --- TRAIN grid search ---
        res, valid = grid_search(df[train_mask], z[train_mask], beta[train_mask],
                                 select_costs, CFG.entry_grid, CFG.exit_grid, CFG.min_trades)
        res.to_csv(f"{args.out_dir}/grid_{method}.csv", index=False)

        if valid.empty:
            print(f"  UYARI: en az {CFG.min_trades} trade eden kombinasyon yok, atlaniyor")
            continue

        top = valid.sort_values("total_return_pct", ascending=False)
        print(f"\n  TRAIN en iyi 5 ({len(valid)}/{len(res)} gecerli kombinasyon):")
        print(top[["entry_z", "exit_z", "total_trades", "win_rate_pct",
                   "total_return_pct", "max_drawdown_pct"]].head().to_string(index=False))

        best = top.iloc[0]
        e, x = float(best["entry_z"]), float(best["exit_z"])
        chosen[method] = (e, x)

        edge = []
        if e in (CFG.entry_grid[0], CFG.entry_grid[-1]):
            edge.append("entry_z")
        if x in (CFG.exit_grid[0], CFG.exit_grid[-1]):
            edge.append("exit_z")
        warn = f"  <-- UYARI: {', '.join(edge)} grid sinirinda" if edge else ""
        print(f"\n  SECILEN: entry_z={e}  exit_z={x}  (train {best['total_return_pct']:+.2f}%, "
              f"{int(best['total_trades'])} trade){warn}")

        # --- TEST (out-of-sample) ---
        print(f"\n  TEST (out-of-sample):")
        for slip in CFG.report_slippage_bps:
            c = Costs(initial_capital=CFG.initial_capital, slippage_bps=slip,
                      spread_cents=CFG.spread_cents, commission_per_fill=CFG.commission_per_fill)
            trades, equity = run_backtest(df[test_mask], z[test_mask], beta[test_mask], e, x, c)
            s = summarize(trades, equity, CFG.initial_capital, label=f"{method}_slip{slip:g}")
            s.update(method=method, entry_z=e, exit_z=x, slippage_bps=slip)
            oos_rows.append(s)
            print(f"\n  --- slipaj {slip:g}/10000 ---")
            print(format_summary(s))
            if slip == CFG.select_slippage_bps and len(trades):
                trades.to_csv(f"{args.out_dir}/trades_{method}.csv", index=False)
                equity.to_csv(f"{args.out_dir}/equity_{method}.csv")
                yearly[method] = yearly_pnl(trades)

    if not oos_rows:
        print("\nHicbir yontem gecerli sonuc uretmedi.")
        return

    oos = pd.DataFrame(oos_rows).set_index("label")
    oos.to_csv(f"{args.out_dir}/oos_results.csv")

    print(f"\n{'='*70}\nOZET - TEST donemi getirisi (%)\n{'='*70}")
    pivot = oos.pivot_table(index="method", columns="slippage_bps", values="total_return_pct")
    pivot.columns = [f"slip {c:g}/10000" for c in pivot.columns]
    print(pivot.round(2).to_string())

    if yearly:
        yr = pd.DataFrame(yearly)
        yr.to_csv(f"{args.out_dir}/yearly_pnl.csv")
        print(f"\nYillara gore net PnL ($, slipaj {CFG.select_slippage_bps:g}/10000):")
        print(yr.round(0).to_string())

    with open(f"{args.out_dir}/meta.json", "w") as f:
        json.dump({
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in CFG.__dict__.items()},
            "chosen_params": chosen,
            "train_range": [str(df.index[train_mask].min()), str(df.index[train_mask].max())],
            "test_range": [str(df.index[test_mask].min()), str(df.index[test_mask].max())],
        }, f, indent=2, default=str)

    print(f"\nKaydedildi -> {args.out_dir}/")


if __name__ == "__main__":
    sys.exit(main())
