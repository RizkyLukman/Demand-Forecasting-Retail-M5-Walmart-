# Hardware profile (Phase 1)

Measured on 2026-10-22/23. Raw numbers: `reports/hardware_profile.json`, `reports/phase1_io.json`.

## Machine

| Item | Value |
|---|---|
| CPU | AMD64 Family 25 Model 68 (16 logical cores) |
| RAM | 15.3 GB total; **only 3–5 GB available** during the runs (Chrome, VS Code, Discord open) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, driver 610.74 |
| Software | Python 3.12.5, xgboost 3.4.1 (CUDA build), pandas 2.3.3, numpy 2.5.4 |

RAM figures below are **peak RSS of the Python process** (psutil, sampled every 0.05–0.2 s).
VRAM is device-wide `memory.used` from `nvidia-smi` (polled every 0.5 s), reported as
peak minus the value at start.

## 1. Load, downcast, long format (`python -m retailfc.io`)

| Step | Result |
|---|---|
| Load all three CSVs (int16 / float32 / category) | 3.3 s, peak RSS 0.29 GB |
| Long format, per store, all 10 stores | 1.4–1.5 s per store |
| Peak RSS, whole run | **0.51 GB** |
| Rows kept (after dropping days before first sale) | 46,796,220 |
| In-memory size, all stores | 672 MB |
| Parquet on disk, all stores | 99.8 MB |

The pre-run estimate (1–2 GB long table, melt at 2–3×) was too high: per-store numpy
conversion avoids `pandas.melt` entirely.

## 2. Training one store (`python -m retailfc.hardware_profile --store CA_1`)

Setup: store CA_1, cutoff d_1885, train d ≤ 1857 (4,525,448 rows), early-stopping set
d_1858–d_1885 (85,372 rows). **27 provisional features** (lags, rolling stats, calendar,
price, `item_id`/`dept_id`/`cat_id` as pandas categories). `QuantileDMatrix`,
`tree_method="hist"`, `max_bin=256`, `eta=0.05`, `max_depth=8`, early stopping 100 rounds.

| Step | Time | Peak RSS | VRAM (delta) |
|---|---|---|---|
| Provisional features | 5.1 s | 1.0 GB | – |
| QuantileDMatrix train + valid | 6.1 s | 3.0 GB | – |
| `reg:tweedie` (p=1.2) | 11.8 s, 225 rounds (best 124), 0.053 s/round | 2.3 GB | +514 MiB (peak 577) |
| `reg:quantileerror`, 4 quantiles in one model | 632 s, 3000 rounds, **no early stop**, 0.211 s/round | **10.5 GB** | +790 MiB (peak 1,097) |

## 3. Why the quantile run peaked at 10.5 GB

Investigated, not guessed (diagnostic scripts kept outside the repo):

1. RSS during boosting grows slowly: 2.30 → 2.47 GB over 1,000 rounds.
2. The spike happens **at the end of `xgb.train`**. In xgboost 3.x, `train()` returns
   `bst.reset()`, which releases training caches by re-serialising the model. Peak RSS
   for a 1,000-round run was 7.4 GB, while RSS right after `train()` returned was 2.36 GB.
3. The spike scales with **model size**, and model size is driven by `item_id` as a
   categorical feature (100 rounds each, same data):

| Config | Trees | Serialized model | Peak RSS |
|---|---:|---:|---:|
| tweedie, `item_id` categorical | 100 | 48.3 MB | 4.80 GB |
| 4 quantiles, `item_id` categorical | 400 | 283.9 MB | 5.58 GB |
| 4 quantiles, categorical + `max_cat_threshold=64` | 400 | 283.9 MB | 5.64 GB |
| 4 quantiles, `item_id` **label-encoded** | 400 | **6.9 MB** | 4.78 GB |
| 1 quantile (p50), `item_id` categorical | 100 | 75.5 MB | 4.92 GB |

(Baseline RSS in this diagnostic was ~4.8 GB because two DMatrix variants were held at once.)

A 1,000-round, 4-quantile model with categorical `item_id` serialises to **1.6 GB**.

## 4. Conclusions for later phases

- **VRAM is not the bottleneck.** One store uses < 1.1 GB of the 4 GB, for both objectives.
  The pre-run estimate (0.5–1 GB per store) holds.
- **Host RAM is the bottleneck**, through model size, not data size.
- **Phase 5 decision input:** native categorical `item_id` makes trees ~40× larger than
  label encoding. The plan already requires comparing the two; model size and peak RAM
  must be part of that comparison, not only accuracy.
- **Quantile training is slow to converge** at `eta=0.05` (3,000 rounds, no early stop,
  10.5 min per store). Ten stores × four folds would take ~7 hours. Needs a higher
  learning rate or a round cap for the quantile model, decided in Phase 5.
- Run heavy training with other applications closed: 3–5 GB free RAM is not enough
  margin for a 10 GB spike.
- Re-run this profile in Phase 4 with the final feature set (the plan estimates ~60
  features; this profile used 27).
