# Profil hardware (Fase 1)

Diukur pada 2026-10-22/23. Angka mentah: `reports/hardware_profile.json`, `reports/phase1_io.json`.

## Mesin

| Item | Nilai |
|---|---|
| CPU | AMD64 Family 25 Model 68 (16 core logis) |
| RAM | Total 15,3 GB; **hanya 3–5 GB tersedia** selama pengukuran (Chrome, VS Code, Discord terbuka) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4096 MiB, driver 610.74 |
| Software | Python 3.12.5, xgboost 3.4.1 (build CUDA), pandas 2.3.3, numpy 2.5.4 |

Angka RAM di bawah adalah **puncak RSS proses Python** (psutil, sampel tiap 0,05–0,2 detik).
VRAM adalah `memory.used` seluruh device dari `nvidia-smi` (dibaca tiap 0,5 detik),
dilaporkan sebagai puncak dikurangi nilai di awal.

## 1. Muat, downcast, format panjang (`python -m retailfc.io`)

| Langkah | Hasil |
|---|---|
| Muat ketiga CSV (int16 / float32 / category) | 3,3 detik, puncak RSS 0,29 GB |
| Format panjang, per toko, 10 toko | 1,4–1,5 detik per toko |
| Puncak RSS, seluruh proses | **0,51 GB** |
| Baris disimpan (setelah buang hari sebelum penjualan pertama) | 46.796.220 |
| Ukuran di memori, semua toko | 672 MB |
| Parquet di disk, semua toko | 99,8 MB |

Estimasi sebelum pengukuran (tabel panjang 1–2 GB, melt 2–3×) terlalu tinggi: konversi
per toko dengan numpy sama sekali tidak memakai `pandas.melt`.

## 2. Training satu toko (`python -m retailfc.hardware_profile --store CA_1`)

Setup: toko CA_1, cutoff d_1885, train d ≤ 1857 (4.525.448 baris), set early stopping
d_1858–d_1885 (85.372 baris). **27 fitur sementara** (lag, statistik rolling, kalender,
harga, `item_id`/`dept_id`/`cat_id` sebagai kategori pandas). `QuantileDMatrix`,
`tree_method="hist"`, `max_bin=256`, `eta=0.05`, `max_depth=8`, early stopping 100 iterasi.

| Langkah | Waktu | Puncak RSS | VRAM (selisih) |
|---|---|---|---|
| Fitur sementara | 5,1 detik | 1,0 GB | – |
| QuantileDMatrix train + valid | 6,1 detik | 3,0 GB | – |
| `reg:tweedie` (p=1,2) | 11,8 detik, 225 iterasi (terbaik 124), 0,053 detik/iterasi | 2,3 GB | +514 MiB (puncak 577) |
| `reg:quantileerror`, 4 kuantil dalam satu model | 632 detik, 3000 iterasi, **tidak berhenti lebih awal**, 0,211 detik/iterasi | **10,5 GB** | +790 MiB (puncak 1.097) |

## 3. Kenapa model kuantil memuncak di 10,5 GB

Diselidiki dengan pengukuran, bukan tebakan (skrip diagnostik disimpan di luar repo):

1. RSS selama boosting naik pelan: 2,30 → 2,47 GB dalam 1.000 iterasi.
2. Lonjakan terjadi **di akhir `xgb.train`**. Di xgboost 3.x, `train()` mengembalikan
   `bst.reset()`, yang melepas cache training dengan menserialisasi ulang model. Puncak
   RSS untuk run 1.000 iterasi adalah 7,4 GB, sedangkan RSS tepat setelah `train()`
   selesai hanya 2,36 GB.
3. Lonjakan sebanding dengan **ukuran model**, dan ukuran model ditentukan oleh `item_id`
   sebagai fitur kategorikal (masing-masing 100 iterasi, data sama):

| Konfigurasi | Pohon | Model terserialisasi | Puncak RSS |
|---|---:|---:|---:|
| tweedie, `item_id` kategorikal | 100 | 48,3 MB | 4,80 GB |
| 4 kuantil, `item_id` kategorikal | 400 | 283,9 MB | 5,58 GB |
| 4 kuantil, kategorikal + `max_cat_threshold=64` | 400 | 283,9 MB | 5,64 GB |
| 4 kuantil, `item_id` **label encoding** | 400 | **6,9 MB** | 4,78 GB |
| 1 kuantil (p50), `item_id` kategorikal | 100 | 75,5 MB | 4,92 GB |

(RSS dasar di diagnostik ini ~4,8 GB karena dua varian DMatrix disimpan sekaligus.)

Model 4 kuantil 1.000 iterasi dengan `item_id` kategorikal berukuran **1,6 GB** setelah diserialisasi.

## 3b. Profil ulang di Fase 4 dengan fitur final

`python -m retailfc.hardware_profile --store CA_1 --features final`
(angka: `reports/hardware_profile_phase4.json`). Setup sama dengan bagian 2 (CA_1, cutoff
d_1885, parameter sama), tetapi memakai **36 fitur** dari tabel dasar Fase 4 dan
mengeluarkan hari tutup dari baris train. RAM tersedia saat mulai: 7,6 GB. (Fitur ke-37,
`zero_run_lag28`, ditambahkan sesudahnya; satu kolom float32 tambahan tidak mengubah
kesimpulan memori.)

| Langkah / model | `item_id` | Waktu | Iterasi (terbaik) | Model | Puncak RSS | VRAM (selisih) |
|---|---|---|---|---:|---:|---:|
| Muat tabel dasar + fitur per fold (523 MB) | – | – | – | – | 2,16 GB | – |
| `reg:tweedie` | kategorikal | 17,2 s | 284 (183) | 86,2 MB | 4,58 GB | +562 MiB |
| `reg:tweedie` | label | 14,4 s | 240 (139) | **3,7 MB** | 4,29 GB | +452 MiB |
| 4 kuantil | label | 574,7 s | 3000 (2993) | 100,3 MB | **4,96 GB** | +774 MiB |

- Dengan label encoding, puncak RAM model 4 kuantil turun dari **10,5 GB (Fase 1) ke 5,0 GB**,
  meskipun fiturnya lebih banyak (36 vs 27).
- Untuk tweedie, label encoding membuat model 23× lebih kecil (3,7 vs 86,2 MB).
- **Masih terbuka:** model kuantil tetap tidak berhenti lebih awal di 3.000 iterasi
  (~10 menit per toko per fold). Perlu `eta` lebih tinggi atau batas iterasi di Fase 5.

## 4. Kesimpulan untuk fase berikutnya

- **VRAM bukan hambatan.** Satu toko memakai < 1,1 GB dari 4 GB, untuk kedua objective.
  Estimasi awal (0,5–1 GB per toko) terbukti.
- **RAM host yang jadi hambatan**, lewat ukuran model, bukan ukuran data.
- **Masukan untuk keputusan Fase 5:** `item_id` kategorikal membuat pohon ~40× lebih besar
  daripada label encoding. Rencana proyek sudah mewajibkan perbandingan keduanya; ukuran
  model dan puncak RAM harus ikut dibandingkan, bukan hanya akurasi.
- **Training kuantil lambat konvergen** di `eta=0.05` (3.000 iterasi tanpa early stop,
  10,5 menit per toko). 10 toko × 4 fold akan makan ~7 jam. Butuh learning rate lebih
  tinggi atau batas iterasi khusus model kuantil, diputuskan di Fase 5.
- Jalankan training berat dengan aplikasi lain ditutup: RAM bebas 3–5 GB tidak cukup
  untuk lonjakan 10 GB.
- Jalankan ulang profil ini di Fase 4 dengan set fitur final (rencana memperkirakan
  ~60 fitur; profil ini memakai 27).
