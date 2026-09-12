"""Veri yukleme ve hizalama."""

import numpy as np
import pandas as pd
import warnings

from config import CFG


def load_pair(path: str = None, resample: str = None,
              regular_hours_only: bool = None, clean: bool = False) -> pd.DataFrame:
    """
    V ve MA kapanis fiyatlarini tek bir DataFrame'de hizalar.

    Donen DataFrame: DatetimeIndex (UTC), kolonlar ['price_v', 'price_ma'].
    Varsayilan, fiyat sicramalarini korur. clean=True yalnizca geriye donuk
    veri incelemesi icindir: sonraki bara bakarak veri siler.
    """
    path = path or CFG.data_path
    resample = resample if resample is not None else CFG.resample
    regular_hours_only = CFG.regular_hours_only if regular_hours_only is None else regular_hours_only

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    wide = df.pivot_table(index="timestamp", columns="ticker", values="close")
    wide = wide.rename(columns={"V": "price_v", "MA": "price_ma"})
    wide = wide[["price_v", "price_ma"]].dropna()

    if regular_hours_only:
        # pre-market / after-hours barlari cok ince; tek-bar fiyat sicramalari
        # uretip sinyalde yapay outlier'a yol aciyor
        local = wide.index.tz_convert("America/New_York")
        mask = ((local.time >= pd.Timestamp("09:30").time()) &
                (local.time <= pd.Timestamp("16:00").time()))
        wide = wide[mask]

    if resample:
        wide = wide.resample(resample).last().dropna()

    if clean:
        wide = clean_bad_prints(wide)

    return wide


def clean_bad_prints(df: pd.DataFrame, jump: float = 0.015,
                     revert: float = 0.6, verbose: bool = True) -> pd.DataFrame:
    """
    Tek-bar bozuk print temizligi.

    Imza: bar t'de buyuk bir siçrama (|log getiri| > jump) ve hemen ardindan
    t+1'de bunun cogunu (>revert orani) geri alan ters yonlu hareket.
    Bu desen tek basina veri hatasini kanitlamaz; gercek hareketler de geri
    donebilir. t+1 kullanildigi icin yalnizca retrospektif tani amaclidir.

    Neden onemli: beta = cov/var oldugu icin TEK bir aykiri cift butun
    pencereyi bozar. Olculdu (2023-01-24 acilis bari): o bari iceren
    pencerelerde beta medyani -0.818 (%100 negatif), hemen oncesindeki
    esit donemde +0.511 (%0 negatif). Yani tek bar, 1 hafta boyunca hedge
    oranini ters cevirir.

    Bozuk bar SILINIR; boylece t-1 -> t+1 getirisi dogrudan hesaplanir ve
    gercek net hareket korunur, sahte sicrama yok olur.
    """
    warnings.warn(
        "Retrospective cleaner uses future prices (t+1); do not use it "
        "to claim a causal backtest. Price reversals need not be bad prints.",
        UserWarning, stacklevel=2,
    )
    bad = np.zeros(len(df), dtype=bool)
    for col in ("price_v", "price_ma"):
        r = np.log(df[col].to_numpy())
        r = np.concatenate([[0.0], np.diff(r)])
        nxt = np.roll(r, -1); nxt[-1] = 0.0
        hit = (np.abs(r) > jump) & (np.sign(nxt) == -np.sign(r)) & (np.abs(nxt) > revert * np.abs(r))
        bad |= hit

    if verbose and bad.any():
        for ts in df.index[bad]:
            print(f"  bozuk print atildi: {ts}")
    return df[~bad]


def train_test_masks(index: pd.DatetimeIndex, split_date: str = None):
    """Train = split oncesi, Test = split ve sonrasi. numpy bool array doner."""
    split_date = split_date or CFG.split_date
    train = (index < split_date).to_numpy() if hasattr(index < split_date, "to_numpy") else np.asarray(index < split_date)
    return train, ~train


def session_gap_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Her seansin ILK bari icin True. O bardaki getiri, onceki seansin
    kapanisindan bu seansin acilisina olan gecelik bosluktur - dakikalik
    bir hareket degildir.
    """
    d = index.tz_convert("America/New_York").date
    m = np.zeros(len(index), dtype=bool)
    m[0] = True
    m[1:] = d[1:] != d[:-1]
    return m


def bars_per_year(resample: str = None) -> int:
    """Yaklasik: 252 islem gunu x 390 dakika / bar dakikasi."""
    resample = resample or CFG.resample
    minutes = pd.Timedelta(resample).total_seconds() / 60
    return int(252 * 390 / minutes)
