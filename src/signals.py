"""
Iki BAGIMSIZ sinyal uretici. Her biri kendi beta'sini (hedge ratio) ve
kendi z-score'unu (mispricing) uretir; biri digerinin ciktisini kullanmaz.

Ikisi de causal: t anindaki deger yalnizca t ve oncesindeki barlari kullanir.

  A) KALMAN  - V_t = alpha_t + beta_t*MA_t, katsayilar random walk.
               Fiyat SEVIYESI uzerinde calisir; bu kisa pencerede sahte
               regresyona yol acardi, ancak Kalman tum gecmisi bir random-walk
               prior'i ile biriktirdigi icin beta stabil kalir (~0.6).

  B) WAVELET - beta, log GETIRILER uzerinden hesaplanir. Seviye uzerinde
               kisa pencere regresyonu sahte regresyon uretiyordu (V ve MA
               ikisi de random walk oldugundan bir haftalik pencerede seviye
               korelasyonu negatife donebiliyor, olculdu: -0.60 seviye vs
               +0.47 getiri). Getiriler yaklasik duragan oldugu icin bu sorun
               ortadan kalkar. Wavelet denoising de getiri serisine uygulanir
               ("duragan sinyal + gurultu" varsayimina uyar).
"""

import numpy as np
import pandas as pd
import pywt

from config import CFG


# --------------------------------------------------------------------------
# ortak yardimci
# --------------------------------------------------------------------------

def wavelet_denoise(series: np.ndarray, wavelet: str = None) -> np.ndarray:
    """DWT + VisuShrink soft-threshold + yeniden insa."""
    wavelet = wavelet or CFG.wavelet
    n = len(series)
    level = min(pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len), 6)

    coeffs = pywt.wavedec(series, wavelet, level=level)
    details = coeffs[1:]

    # en ince olcek detaylarindan MAD ile gurultu std tahmini
    finest = details[-1]
    sigma = np.median(np.abs(finest - np.median(finest))) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(n))

    denoised = [coeffs[0]] + [pywt.threshold(c, value=thresh, mode="soft") for c in details]
    return pywt.waverec(denoised, wavelet)[:n]


# --------------------------------------------------------------------------
# A) Kalman
# --------------------------------------------------------------------------

def kalman_signal(price_v: np.ndarray, price_ma: np.ndarray,
                  delta: float = None, ve: float = None):
    """
    Return: beta (n,), z (n,)
      beta_t : hedge ratio
      z_t    : olcum artiginin standardize hali = e_t / sqrt(Q_t)
               (Q_t, Kalman'in kendi tahmin ettigi artik varyansi)
    """
    delta = CFG.kalman_delta if delta is None else delta
    ve = CFG.kalman_ve if ve is None else ve

    n = len(price_v)
    vw = delta / (1 - delta) * np.eye(2)

    theta = np.zeros(2)          # [alpha, beta]
    P = np.zeros((2, 2))

    beta = np.zeros(n)
    z = np.zeros(n)

    for t in range(n):
        R = P + vw if t > 0 else P
        H = np.array([1.0, price_ma[t]])

        e = price_v[t] - H @ theta       # olcum artigi
        Q = H @ R @ H.T + ve             # artigin varyansi

        K = (R @ H) / Q
        theta = theta + K * e
        P = R - np.outer(K, H) @ R

        beta[t] = theta[1]
        z[t] = e / np.sqrt(Q)

    return beta, z


# --------------------------------------------------------------------------
# B) Wavelet
# --------------------------------------------------------------------------

def wavelet_signal(price_v: np.ndarray, price_ma: np.ndarray,
                   window: int = None, wavelet: str = None,
                   beta_clip: tuple = (0.0, 2.0), beta_valid: tuple = None,
                   gap_mask: np.ndarray = None):
    """
    Return: beta (n,), z (n,)

    beta_t : [t-window, t) araligindaki log getiriler wavelet ile denoise
             edilip cov/var ile hesaplanir. Negatif beta ekonomik anlam
             tasimadigindan [0, 2] araligina kirpilir.

    z_t    : spread_k = log(V_k) - beta_t*log(MA_k) serisi ayni pencerede
             SABIT beta_t ile kurulur (beta zamanla degisirken spread'e
             yapay ziplama girmesin diye), pencerenin kendi ortalama/std'si
             ile z-score'lanir.
    """
    window = window or CFG.window
    wavelet = wavelet or CFG.wavelet
    beta_valid = CFG.beta_valid if beta_valid is None else beta_valid

    n = len(price_v)
    log_v = np.log(price_v)
    log_ma = np.log(price_ma)

    r_v = np.zeros(n); r_v[1:] = np.diff(log_v)
    r_ma = np.zeros(n); r_ma[1:] = np.diff(log_ma)

    # Seans acilis barinin "getirisi" aslinda GECELIK BOSLUK (~17.5 saat),
    # dakikalik bir hareket degil. Buyuklugu dakikalik getirilerin onlarca
    # kati oldugundan cov/var hesabinda orantisiz agirlik kazanir: olculdu,
    # 1950 barlik bir pencerede 4 acilis bari kovaryansa -4.7e-05 katki
    # yaparken pencerenin net kovaryansi +1.0e-07 idi (~455 kat) ve beta'yi
    # negatife cevirmisti. Bu getiriler 0'lanarak regresyondan cikarilir.
    if gap_mask is not None:
        r_v = np.where(gap_mask, 0.0, r_v)
        r_ma = np.where(gap_mask, 0.0, r_ma)

    beta = np.zeros(n)

    for t in range(window, n):
        rv_d = wavelet_denoise(r_v[t - window:t], wavelet)
        rm_d = wavelet_denoise(r_ma[t - window:t], wavelet)

        var_m = rm_d.var(ddof=1)
        if var_m <= 0:
            continue

        beta[t] = float(np.clip(np.cov(rv_d, rm_d, ddof=1)[0, 1] / var_m, *beta_clip))

    # --- z-score ---
    # s_k = log_v[k] - b_t*log_ma[k] icin pencere istatistikleri, b_t sabit
    # tutularak analitik olarak acilir (her barda dilim almak yerine):
    #   mean(s) = m_v - b*m_ma
    #   var(s)  = var_v + b^2*var_ma - 2*b*cov(v,ma)
    sv = pd.Series(log_v)
    sm = pd.Series(log_ma)
    roll_v = sv.rolling(window)
    roll_m = sm.rolling(window)

    m_v = roll_v.mean().shift(1).to_numpy()
    m_ma = roll_m.mean().shift(1).to_numpy()
    var_v = roll_v.var(ddof=1).shift(1).to_numpy()
    var_ma = roll_m.var(ddof=1).shift(1).to_numpy()
    cov_vm = sv.rolling(window).cov(sm, ddof=1).shift(1).to_numpy()

    var_s = var_v + beta**2 * var_ma - 2.0 * beta * cov_vm
    sd = np.sqrt(np.where(var_s > 0, var_s, np.nan))
    mean_s = m_v - beta * m_ma

    z = (log_v - beta * log_ma - mean_s) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z[beta == 0] = 0.0

    return beta, z


SIGNALS = {
    "kalman": kalman_signal,
    "wavelet": wavelet_signal,
}
