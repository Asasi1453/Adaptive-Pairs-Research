"""
Backtest motoru: z-score esikli mean-reversion, gercekci maliyetler,
beta-volatilitesine gore pozisyon boyutlandirma.

Pozisyon:
   +1 long spread  : V al, beta kadar MA sat
   -1 short spread : V sat, beta kadar MA al
    0 flat

Her round-trip 4 fill uretir (giris V + giris MA + cikis V + cikis MA).
Her fill'de fiyat aleyhe kayar: mid +/- (mid*slipaj + spread/2).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CFG


@dataclass(frozen=True)
class Costs:
    initial_capital: float = CFG.initial_capital
    slippage_bps: float = CFG.select_slippage_bps   # 10000'de kac birim
    spread_cents: float = CFG.spread_cents
    commission_per_fill: float = CFG.commission_per_fill


def positions_from_z(z: np.ndarray, entry_z: float, exit_z: float,
                     entry_mask: np.ndarray = None) -> np.ndarray:
    """
    |z| entry_z'yi asinca pozisyon ac, exit_z'nin altina inince kapat.

    entry_mask: True olan barlarda YENI GIRIS'e izin verilir. False barlar
        yalnizca girisi engeller - acik pozisyon kapatilmaz. (Pozisyonu
        zorla kapatmak, kosul duzelince yeniden acilmasina ve gereksiz
        round-trip maliyetine yol aciyordu.)
    """
    n = len(z)
    pos = np.zeros(n, dtype=np.int8)
    cur = 0
    for t in range(n):
        if cur == 0:
            if entry_mask is None or entry_mask[t]:
                if z[t] > entry_z:
                    cur = -1
                elif z[t] < -entry_z:
                    cur = 1
        elif abs(z[t]) <= exit_z:
            cur = 0
        pos[t] = cur
    return pos


def _fill(mid: float, is_buy: bool, slip_rate: float, half_spread: float) -> float:
    cost = mid * slip_rate + half_spread
    return mid + cost if is_buy else mid - cost


def _size_factors(beta: np.ndarray, vol_window: int, lo: float, hi: float) -> np.ndarray:
    """
    Beta'nin rolling volatilitesine gore sermaye kullanim orani.
    Beta normalden oynaksa (hedge orani guvenilmezse) daha az sermaye kullan.
    Referans = medyan beta-volatilitesi. Kaldirac yok (ust sinir hi).
    """
    vol = pd.Series(beta).rolling(vol_window, min_periods=30).std().to_numpy()
    ref = np.nanmedian(vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((vol > 0) & np.isfinite(vol) & np.isfinite(ref), ref / vol, 1.0)
    return np.clip(np.nan_to_num(f, nan=1.0, posinf=hi), lo, hi)


def run_backtest(df: pd.DataFrame, z: np.ndarray, beta: np.ndarray,
                 entry_z: float, exit_z: float, costs: Costs = Costs(),
                 exec_lag: int = 1, entry_mask: np.ndarray = None):
    """
    df: 'price_v', 'price_ma' kolonlu, DatetimeIndex'li.

    exec_lag: sinyal ile emir arasindaki bar gecikmesi. z_t bar t'nin
        KAPANIS fiyatindan hesaplandigi icin o fiyattan islem yapmak
        gerceklesemez (sinyali gorup ayni anda o fiyattan almak mumkun
        degil). exec_lag=1 ile pozisyon bir bar kaydirilir: t barinda
        uretilen sinyal t+1 barinin fiyatindan uygulanir.

    Return: trades (DataFrame), equity (Series)
    """
    price_v = df["price_v"].to_numpy()
    price_ma = df["price_ma"].to_numpy()
    index = df.index
    n = len(price_v)

    pos = positions_from_z(z, entry_z, exit_z, entry_mask)
    if exec_lag:
        pos = np.concatenate([np.zeros(exec_lag, dtype=pos.dtype), pos[:-exec_lag]])
    size_f = _size_factors(beta, CFG.vol_window, CFG.min_size_factor, CFG.max_size_factor)

    slip_rate = costs.slippage_bps / 10_000.0
    half_spread = (costs.spread_cents / 100.0) / 2.0
    commission_rt = costs.commission_per_fill * 4

    capital = costs.initial_capital
    trades = []

    entry_i = None
    cur = 0

    for t in range(n + 1):                     # n. adim: acik pozisyonu kapat
        p = pos[t] if t < n else 0

        if cur == 0 and p != 0:
            entry_i, cur = t, p
            continue

        if cur != 0 and p != cur:
            close_i = min(t, n - 1)
            direction = cur

            notional = capital * size_f[entry_i]
            b = beta[entry_i]
            ev, em = price_v[entry_i], price_ma[entry_i]
            denom = ev + abs(b) * em
            if denom <= 0:
                cur, entry_i = 0, None
                continue

            sh_v = notional / denom
            sh_ma = abs(b) * sh_v
            xv, xm = price_v[close_i], price_ma[close_i]

            if direction == 1:                 # long spread: V al, MA sat
                ev_f = _fill(ev, True, slip_rate, half_spread)
                em_f = _fill(em, False, slip_rate, half_spread)
                xv_f = _fill(xv, False, slip_rate, half_spread)
                xm_f = _fill(xm, True, slip_rate, half_spread)
                gross = sh_v * (xv_f - ev_f) + sh_ma * (em_f - xm_f)
            else:                              # short spread: V sat, MA al
                ev_f = _fill(ev, False, slip_rate, half_spread)
                em_f = _fill(em, True, slip_rate, half_spread)
                xv_f = _fill(xv, True, slip_rate, half_spread)
                xm_f = _fill(xm, False, slip_rate, half_spread)
                gross = sh_v * (ev_f - xv_f) + sh_ma * (xm_f - em_f)

            slip_cost = (sh_v * (abs(ev_f - ev) + abs(xv_f - xv)) +
                         sh_ma * (abs(em_f - em) + abs(xm_f - xm)))
            net = gross - commission_rt
            capital += net

            trades.append(dict(
                entry_time=index[entry_i], exit_time=index[close_i],
                direction="long" if direction == 1 else "short",
                bars_held=close_i - entry_i,
                size_factor=size_f[entry_i], notional=notional, beta=b,
                gross_pnl=gross, slippage_spread_cost=slip_cost,
                commission=commission_rt, net_pnl=net,
                return_pct=net / notional * 100 if notional else np.nan,
                capital_after=capital,
            ))

            cur, entry_i = 0, None
            if t < n and p != 0:
                entry_i, cur = t, p

    trades_df = pd.DataFrame(trades)

    if len(trades_df):
        eq = pd.concat([
            pd.Series([costs.initial_capital], index=[index[0]]),
            pd.Series(trades_df["capital_after"].to_numpy(), index=trades_df["exit_time"]),
        ])
        equity = eq[~eq.index.duplicated(keep="last")].sort_index()
    else:
        equity = pd.Series([costs.initial_capital], index=[index[0]])

    return trades_df, equity
