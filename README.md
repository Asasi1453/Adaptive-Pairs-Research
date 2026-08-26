# V / MA Pairs Trading — Kalman & Wavelet

Visa (V) ve Mastercard (MA) çifti üzerinde pairs trading araştırması. İki **bağımsız**
hedge-ratio tahmin yöntemi karşılaştırılıyor: Kalman filter ve wavelet transform.

Veri: [mito0o852/OHLCV-1m](https://huggingface.co/datasets/mito0o852/OHLCV-1m) —
dakikalık OHLCV, 2021-01 → 2026-03, sadece normal seans (09:30–16:00 ET).

## Yöntem

**Kalman** — `V_t = α_t + β_t·MA_t`, katsayılar random walk. Fiyat seviyeleri üzerinde
çalışır; tüm geçmişi bir prior ile biriktirdiği için β stabil kalır.

**Wavelet** — β, **log getiriler** üzerinden hesaplanır (seviyeler üzerinde kısa pencere
regresyonu sahte regresyon üretiyordu). Getiriler wavelet (db4, VisuShrink soft-threshold)
ile denoise edilir, ardından `spread = log(V) − β·log(MA)` kurulup pencerenin kendi
ortalama/std'siyle z-score'lanır.

Sinyal: `|z| > entry_z` iken pozisyon açılır, `|z| ≤ exit_z` iken kapatılır.

## Gerçekçilik önlemleri

Backtest sonucunu ciddi şekilde etkileyen, sırayla bulunup düzeltilen konular:

| Konu | Etki |
|---|---|
| **Emir gecikmesi** — sinyal bar kapanışında üretilip aynı kapanıştan işlem yapılamaz; 1 bar kaydırıldı | edge'in ~yarısı |
| **Bozuk veri printi** — 2023-01-24 açılışında V/MA ters yönde hatalı fiyat; `cov/var` tek aykırı çifte duyarlı olduğundan β bir hafta boyunca −0.82'ye çakılıyordu | negatif β %0.67 → %0.26 |
| **Gecelik boşluklar** — açılış barının "1 dakikalık getirisi" aslında ~17.5 saatlik hareket; kovaryansta orantısız ağırlık kazanıyordu | negatif β %0.26 → **%0.00** |
| **Maliyetler** — komisyon ($1.5/fill × 4), 1¢ spread, slipaj senaryoları | ham edge'in büyük kısmı |
| **Train/test ayrımı** — parametre yalnızca train'de seçilir, sonuç yalnızca test'te ölçülür | overfitting'i açığa çıkardı |
| **β sağlık filtresi** — β ∉ [0.2, 1.5] iken yeni giriş yok (pozisyon tutulur; zorla kapatmak churn üretiyordu) | — |

## Kurulum

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Veri `data/processed/pair_V_MA.parquet` altında beklenir (repoda yok, `.gitignore`).

## Çalıştırma

```bash
cd src

# 5-dakikalık, train/test protokolü, grid search + out-of-sample
python run.py

# 1-dakikalık, sabit eşikler (entry 2.0σ / exit 0.5σ), tüm grafikler
python wavelet_1min.py --out-dir ../results_1min

# ham vs kırpılmış β karşılaştırması, ±1σ/±2σ bantlarıyla
python beta_compare.py --out-dir ../results_1min
```

Tüm parametreler [`src/config.py`](src/config.py) içinde.

## Dosyalar

| Dosya | Görev |
|---|---|
| `src/config.py` | Tüm parametreler |
| `src/data.py` | Yükleme, hizalama, bozuk print temizliği, seans maskesi |
| `src/signals.py` | Kalman ve wavelet sinyalleri (bağımsız) |
| `src/engine.py` | Backtest: maliyetler, pozisyon boyutu, emir gecikmesi |
| `src/metrics.py` | Performans metrikleri |
| `src/run.py` | 5-dk train/test protokolü |
| `src/wavelet_1min.py` | 1-dk analiz + grafikler |
| `src/beta_compare.py` | Ham vs kırpılmış β |

## Sonuç

1-dakikalık, entry 2.0σ / exit 0.5σ, $100k sermaye, 1113 trade:

| Slipaj | Getiri | Max DD | Win% | PF |
|---|---|---|---|---|
| 1/10000 | +9.14% | −8.64% | 43.4 | 1.13 |
| 2/10000 | **−7.57%** | −15.84% | 37.2 | 0.90 |
| 3/10000 | −21.78% | −25.57% | 32.5 | 0.72 |

Gerçekçi slipaj varsayımı altında strateji **kâr etmiyor**. Erken koşularda görülen
pozitif sonuçlar, sonradan düzeltilen veri/metodoloji hatalarının ürünüydü — özellikle
bozuk hedge oranı ve aynı-bar işlem varsayımı. Ortalama tutuş süresi 195 dk (medyan 11 dk).

Bu bir araştırma çalışmasıdır, yatırım tavsiyesi değildir.
