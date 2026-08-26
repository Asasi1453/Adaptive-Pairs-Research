"""Tek merkezi konfigurasyon - tum kosu parametreleri burada."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # --- veri ---
    data_path: str = "data/processed/pair_V_MA.parquet"
    resample: str = "5min"
    regular_hours_only: bool = True      # 09:30-16:00 ET disi barlar atilir

    # --- donem ayrimi ---
    # parametre secimi SADECE train'de yapilir; test hic gorulmez
    split_date: str = "2023-07-01"

    # --- sermaye & maliyet ---
    initial_capital: float = 100_000.0
    commission_per_fill: float = 1.5     # 4 fill/round-trip => $6/trade
    spread_cents: float = 1.0            # bid-ask; yarisi her fill'de aleyhe
    select_slippage_bps: float = 2.0     # parametre secerken kullanilan senaryo
    report_slippage_bps: tuple = (1.0, 2.0, 3.0)   # raporlanacak senaryolar

    # --- sinyal ---
    window: int = 390                    # rolling pencere (bar) ~ 1 hafta
    wavelet: str = "db4"
    kalman_delta: float = 1e-4
    kalman_ve: float = 1e-3
    # hedge orani bu araligin disindaysa sinyal gecersiz sayilir:
    # beta 0'a cokerse pozisyon fiilen hedge'siz (ciplak V) olur,
    # 1.5 ustunde ise MA bacagi asiri buyur - ikisi de pairs trading degil
    beta_valid: tuple = (0.2, 1.5)

    # --- pozisyon boyutu ---
    vol_window: int = 390
    min_size_factor: float = 0.10        # sermayenin en az %10'u
    max_size_factor: float = 1.00        # kaldirac yok

    # --- grid search ---
    entry_grid: tuple = tuple(round(0.5 + 0.05 * i, 2) for i in range(51))   # 0.50 .. 3.00
    exit_grid: tuple = tuple(round(0.0 + 0.05 * i, 2) for i in range(21))    # 0.00 .. 1.00
    min_trades: int = 20                 # train'de bu sayidan az trade eden kombinasyon gecersiz

    out_dir: str = "results"


CFG = Config()
