"""
HAM (kirpilmamis) beta ile KULLANILAN (kirpilmis) beta karsilastirmasi.

Iki ayri mudahale var, karistirilmamali:

  1) beta_clip = (0.0, 2.0)  -> BETAYI DEGISTIRIR.
     Ham tahmin negatif veya cok buyuk cikabilir; bu araliga kirpilir.

  2) beta_valid = (0.2, 1.5) -> BETAYI DEGISTIRMEZ.
     Yalnizca o barlarda YENI GIRIS acilmasini engeller (acik pozisyon
     tutulmaya devam eder).

Bu script ham betayi (hicbir kirpma/filtre olmadan) hesaplar, kirpilmis
haliyle yan yana cizer ve her ikisine zamana bagli +/-1sigma, +/-2sigma
bantlari ekler (kayan ortalama +/- k * kayan std).
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
from signals import wavelet_denoise

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BLUE, DARKBLUE, BAD, GOOD = "#2a78d6", "#184f95", "#d03b3b", "#0ca30c"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.5, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def raw_wavelet_beta(price_v, price_ma, window, wavelet, gap_mask=None):
    """Hicbir KIRPMA uygulanmamis beta (gecelik bosluklar haric tutulur)."""
    n = len(price_v)
    lv, lm = np.log(price_v), np.log(price_ma)
    r_v = np.zeros(n); r_v[1:] = np.diff(lv)
    r_ma = np.zeros(n); r_ma[1:] = np.diff(lm)
    if gap_mask is not None:
        r_v = np.where(gap_mask, 0.0, r_v)
        r_ma = np.where(gap_mask, 0.0, r_ma)

    beta = np.full(n, np.nan)
    for t in range(window, n):
        rv_d = wavelet_denoise(r_v[t - window:t], wavelet)
        rm_d = wavelet_denoise(r_ma[t - window:t], wavelet)
        var_m = rm_d.var(ddof=1)
        if var_m <= 0:
            continue
        beta[t] = np.cov(rv_d, rm_d, ddof=1)[0, 1] / var_m
    return beta


def sigma_bands(s: pd.Series, window: int):
    m = s.rolling(window, min_periods=window // 4).mean()
    sd = s.rolling(window, min_periods=window // 4).std()
    return m, sd


def _span_label(bars: int) -> str:
    """Bar sayisini islem-gunu/hafta/ay cinsinden okunur etikete cevirir."""
    days = bars / 390.0
    if days >= 19:
        return f"{bars} bar ≈ {days/21:.0f} ay"
    if days >= 4.5:
        return f"{bars} bar ≈ {days/5:.0f} hafta"
    return f"{bars} bar ≈ {days:.0f} gün"


def draw(ax, idx, vals, title, band_window, ylim=None, clip_lines=None, valid_lines=None):
    s = pd.Series(vals, index=idx)
    m, sd = sigma_bands(s, band_window)

    ax.fill_between(idx, (m - 2 * sd).values, (m + 2 * sd).values,
                    color=BLUE, alpha=0.13, linewidth=0, label="±2σ")
    ax.fill_between(idx, (m - sd).values, (m + sd).values,
                    color=BLUE, alpha=0.26, linewidth=0, label="±1σ")

    ax.plot(idx, vals, color=BLUE, lw=0.3, alpha=0.5, label="beta (1-dk)")
    ax.plot(idx, m.values, color=DARKBLUE, lw=1.8,
            label=f"kayan ortalama ({_span_label(band_window)})")

    if clip_lines:
        for y in clip_lines:
            ax.axhline(y, color=BAD, ls="--", lw=1.1)
        ax.plot([], [], color=BAD, ls="--", lw=1.1, label=f"kırpma sınırı {clip_lines}")
    if valid_lines:
        for y in valid_lines:
            ax.axhline(y, color=GOOD, ls=":", lw=1.4)
        ax.plot([], [], color=GOOD, ls=":", lw=1.4, label=f"giriş izni aralığı {valid_lines}")

    ax.axhline(0, color=INK, lw=0.9)
    ax.set_title(title)
    ax.set_ylabel("beta")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/processed/pair_V_MA.parquet")
    ap.add_argument("--out-dir", default="../results_1min")
    ap.add_argument("--window", type=int, default=1950)
    ap.add_argument("--band-window", type=int, default=8190)  # ~1 islem ayi (21g x 390)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_pair(args.data, resample="1min")
    pv, pm = df["price_v"].to_numpy(), df["price_ma"].to_numpy()
    print(f"{len(df):,} bar — ham beta hesaplaniyor...", flush=True)

    cache = f"{args.out_dir}/_raw_beta_w{args.window}_n{len(df)}.npy"
    if os.path.exists(cache):
        raw = np.load(cache)
        print("  (onbellekten okundu)")
    else:
        raw = raw_wavelet_beta(pv, pm, args.window, CFG.wavelet, session_gap_mask(df.index))
        np.save(cache, raw)
    clipped = np.clip(raw, 0.0, 2.0)

    ok = ~np.isnan(raw)
    lo, hi = CFG.beta_valid

    stats = {
        "raw": {
            "median": float(np.nanmedian(raw)), "mean": float(np.nanmean(raw)),
            "std": float(np.nanstd(raw)), "min": float(np.nanmin(raw)), "max": float(np.nanmax(raw)),
            "pct_negative": float(100 * (raw[ok] < 0).mean()),
            "pct_above_2": float(100 * (raw[ok] > 2).mean()),
            "pct_outside_valid": float(100 * ((raw[ok] < lo) | (raw[ok] > hi)).mean()),
            "p01": float(np.nanpercentile(raw, 1)), "p99": float(np.nanpercentile(raw, 99)),
        },
        "clipped": {
            "median": float(np.nanmedian(clipped)), "mean": float(np.nanmean(clipped)),
            "std": float(np.nanstd(clipped)), "min": float(np.nanmin(clipped)), "max": float(np.nanmax(clipped)),
            "pct_at_lower_clip": float(100 * (clipped[ok] <= 1e-12).mean()),
            "pct_at_upper_clip": float(100 * (clipped[ok] >= 2 - 1e-12).mean()),
            "pct_outside_valid": float(100 * ((clipped[ok] < lo) | (clipped[ok] > hi)).mean()),
        },
        "beta_clip": [0.0, 2.0],
        "beta_valid": [lo, hi],
    }
    with open(f"{args.out_dir}/beta_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    r, c = stats["raw"], stats["clipped"]
    print(f"\nHAM beta     : medyan={r['median']:.3f}  ort={r['mean']:.3f}  std={r['std']:.3f}")
    print(f"               min={r['min']:.2f}  max={r['max']:.2f}  (p1={r['p01']:.2f}, p99={r['p99']:.2f})")
    print(f"               NEGATIF={r['pct_negative']:.2f}%   >2 olan={r['pct_above_2']:.2f}%")
    print(f"               {lo}-{hi} disinda={r['pct_outside_valid']:.2f}%")
    print(f"\nKIRPILMIS    : medyan={c['median']:.3f}  ort={c['mean']:.3f}  std={c['std']:.3f}")
    print(f"               alt sinira yapisik={c['pct_at_lower_clip']:.2f}%  ust sinira yapisik={c['pct_at_upper_clip']:.2f}%")
    print(f"               {lo}-{hi} disinda (giris yasak)={c['pct_outside_valid']:.2f}%")

    # ---------- grafik ----------
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    draw(axes[0], df.index, raw,
         f"HAM beta — hiçbir kırpma/filtre yok  "
         f"(negatif %{r['pct_negative']:.2f} · >2 olan %{r['pct_above_2']:.2f} · "
         f"aralık {r['min']:.1f} … {r['max']:.1f})",
         args.band_window,
         ylim=(min(-1.5, np.nanpercentile(raw, 0.2)), max(3.0, np.nanpercentile(raw, 99.8))),
         clip_lines=[0.0, 2.0], valid_lines=[lo, hi])

    draw(axes[1], df.index, clipped,
         f"KULLANILAN beta — [0, 2] aralığına kırpılmış  "
         f"(alt sınıra yapışık %{c['pct_at_lower_clip']:.2f} · üst sınıra yapışık %{c['pct_at_upper_clip']:.2f})",
         args.band_window, ylim=(-0.15, 2.15),
         clip_lines=[0.0, 2.0], valid_lines=[lo, hi])

    fig.suptitle("Hedge ratio (β) — ham vs kırpılmış, zamana bağlı ±1σ / ±2σ bantlarıyla\n"
                 f"σ = {_span_label(args.band_window)} kayan standart sapma  ·  "
                 f"β tahmin penceresi {_span_label(args.window)}",
                 fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{args.out_dir}/beta_raw_vs_clipped.png", dpi=130)
    print(f"\nKaydedildi -> {args.out_dir}/beta_raw_vs_clipped.png, beta_stats.json")


if __name__ == "__main__":
    main()
