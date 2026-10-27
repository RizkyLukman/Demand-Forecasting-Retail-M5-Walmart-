# Project Plan — Demand Forecasting Retail (M5 Walmart → Indonesia) dengan XGBoost

> Dokumen acuan proyek: tujuan, data, batasan hardware, fase kerja, dan aturan.
> Bahasa diskusi, laporan (`reports/`, `data/SOURCES.md`), teks grafik, serta komentar
> dan docstring kode: **Indonesia**. Nama variabel, fungsi, kolom, dan kunci JSON: Inggris.

---

## 1. Tujuan & posisi proyek

**Pertanyaan utama:** Seberapa akurat gradient boosting meramalkan penjualan harian
tingkat produk-toko 28 hari ke depan, di segmen produk mana model gagal, dan apa
dampaknya terhadap keputusan stok? Setelah itu: apakah pipeline yang sama bisa
dipindahkan ke data retail Indonesia dengan efek kalender hijriah (Ramadan/Idulfitri)?

**Model utama: XGBoost** (library `xgboost` ≥ 2.0). Library gradient boosting lain
tidak dipakai di pipeline utama.

**Kenapa proyek ini tidak boleh berhenti di "XGBoost + skor WRMSSE":**
M5 sudah dibahas di ribuan notebook publik. Nilai proyek ini ada pada lima pembeda
berikut. **Semuanya wajib dikerjakan**, bukan opsional:

| # | Pembeda | Fase |
|---|---------|------|
| P1 | Backtesting jujur: beberapa fold *rolling origin*, bukan satu holdout | 3 |
| P2 | Analisis error per segmen (pola permintaan, kategori, event vs non-event) | 6 |
| P3 | Framing bisnis: forecast kuantil → simulasi kebijakan stok (stockout vs overstock) | 7 |
| P4 | Fitur hari libur bergerak, divalidasi dulu di M5 (Easter) sebelum dipakai untuk Lebaran | 4 & 8 |
| P5 | Transfer ke data Indonesia (apotek) dengan fitur Ramadan/Idulfitri | 8 |

---

## 2. Sumber data

### 2.1 Utama — M5 Forecasting (Walmart)
- **Sumber:** Kaggle competition `m5-forecasting-accuracy`
  (`https://www.kaggle.com/competitions/m5-forecasting-accuracy/data`).
  Butuh akun Kaggle dan persetujuan aturan kompetisi. Unduh lewat Kaggle CLI:
  `kaggle competitions download -c m5-forecasting-accuracy`
- **Isi (sudah diverifikasi dari beberapa sumber):** penjualan unit 3.049 produk di 10 toko
  Walmart (CA, TX, WI), 1.941 hari mulai 29 Jan 2011; 3 kategori, 7 departemen;
  30.490 deret di level bawah, 42.840 deret di seluruh level hierarki.
- **File yang diharapkan:** `calendar.csv`, `sell_prices.csv`,
  `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sample_submission.csv`.
  Gunakan **`sales_train_evaluation.csv`** (d_1 … d_1941).
- ⚠️ Nama kolom & rentang tanggal di bawah berasal dari dokumentasi umum.
  **Wajib dicek dari file asli** dan hasil pengecekannya ditulis ke `data/SOURCES.md`:
  - `calendar.csv`: `date, wm_yr_wk, weekday, wday, month, year, d, event_name_1,
    event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI`
  - `sell_prices.csv`: `store_id, item_id, wm_yr_wk, sell_price`
  - sales: `id, item_id, dept_id, cat_id, store_id, state_id, d_1 … d_1941`

### 2.2 Transfer — Data penjualan apotek Indonesia
- **Sumber:** Mendeley Data, "Retail sales dataset of a pharmacy in Indonesia",
  Gustriansyah, R. (2022). DOI `10.17632/2ym7v78wtd.1`. Lisensi **CC BY 4.0**.
  `https://data.mendeley.com/datasets/2ym7v78wtd/1`
- **Isi menurut deskripsi:** tabel `ms_products`, `ms_sales`, `det_sales`, dan satu
  tabel transaksi hasil integrasi. Data berasal dari database apotek yang asli.
- **Hasil audit (28 Okt 2026, `reports/id_data_audit.md`):** satu dump SQL, periode
  1 Jan–31 Des 2015, header nota Jan–Mar sebagian besar hilang (deret lengkap 1 Apr–31 Des,
  275 hari), satu Idulfitri, ±200 produk terjual di ≥ 50% hari. **Tidak memenuhi syarat
  transfer penuh** → jalur "tidak cukup" di Fase 8.
- Diunduh lewat API publik Mendeley ke `data/raw/id_pharmacy/` (checksum di `data/SOURCES.md`).

### 2.3 Kalender Indonesia
- Libur nasional: paket Python `holidays` (mendukung Indonesia, kode `ID`).
  **Verifikasi** tanggal Idulfitri dari paket ini terhadap SKB Libur Nasional/Cuti Bersama
  pemerintah untuk tahun-tahun yang dipakai, karena tanggal Idulfitri bisa berbeda
  1 hari antara perhitungan dan keputusan resmi.
- Simpan hasil final di `data/external/id_calendar.csv` dengan kolom `date, event, source`.
  Kolom `source` wajib diisi (nama dokumen/URL). **Jangan mengisi tanggal dari ingatan.**

### 2.4 Opsional (pemanasan) — Walmart Recruiting: Store Sales Forecasting
- Kompetisi Kaggle lama, data mingguan per departemen untuk 45 toko.
  **Belum diverifikasi ulang**. Cek ketersediaannya dulu. Hanya untuk latihan, tidak
  masuk laporan utama.

### Aturan keras soal data
- **Tidak ada data sintetis/acak** untuk menggantikan data yang belum tersedia.
  Kalau unduhan gagal, berhenti dan laporkan errornya.
- Kredensial (`~/.kaggle/kaggle.json`) **tidak boleh** di-commit, dicetak, atau disalin ke repo.
- Setiap sumber dicatat di `data/SOURCES.md`: URL, tanggal akses, lisensi, ukuran, checksum.

---

## 3. Struktur repo

```
retail-forecast-gbm/
├── PROJECT_PLAN.md
├── README.md                 # ringkasan hasil, bukan tutorial
├── config.yaml               # path, fold, horizon, parameter model, biaya stok
├── pyproject.toml
├── data/
│   ├── SOURCES.md
│   ├── raw/{m5,id_pharmacy}/ # .gitignore
│   ├── external/             # id_calendar.csv
│   ├── interim/              # parquet format panjang, sudah di-downcast
│   └── processed/            # tabel fitur per fold
├── src/retailfc/
│   ├── io.py                 # load + downcast + melt
│   ├── calendar_features.py  # termasuk fitur hari libur bergerak
│   ├── features.py           # lag, rolling, harga, hierarki
│   ├── segments.py           # klasifikasi ADI/CV²
│   ├── folds.py              # rolling origin
│   ├── baselines.py
│   ├── train.py              # XGBoost point + quantile
│   ├── metrics.py            # WRMSSE, RMSSE, MAE, pinball, bias
│   ├── error_analysis.py
│   ├── inventory_sim.py
│   └── transfer_id.py
├── notebooks/                # eksplorasi saja
├── reports/{figures,runs}/
└── tests/
```

---

## 4. Lingkungan & perintah

```powershell
# Windows (PowerShell). Venv sengaja DI LUAR folder proyek: path proyek berisi "→"
# dan numpy/threadpoolctl gagal memuat DLL dari path non-ASCII.
python -m venv $HOME\.venvs\retailfc; & $HOME\.venvs\retailfc\Scripts\Activate.ps1
pip install -e ".[dev]"
# Kaggle CLI >= 2.x: token baru disimpan di ~/.kaggle/access_token (bukan kaggle.json)
# inti: pandas, numpy, pyarrow, xgboost>=2.0, scikit-learn, shap, statsforecast,
#       holidays, matplotlib, pyyaml, kaggle, pytest
# setelah install, catat versi persis: python -c "import xgboost; print(xgboost.__version__)"

python -m retailfc.io          --config config.yaml   # load, downcast, melt -> parquet
python -m retailfc.features    --config config.yaml
python -m retailfc.baselines   --config config.yaml
python -m retailfc.train       --config config.yaml
python -m retailfc.error_analysis --config config.yaml
python -m retailfc.inventory_sim  --config config.yaml
python -m retailfc.transfer_id    --config config.yaml
pytest -q
```

### 4.1 Batasan hardware (WAJIB dipatuhi)

Laptop pengembang: **RAM 16 GB, GPU NVIDIA RTX 3050 Laptop dengan VRAM 4 GB.**

**Angka terukur di Fase 1** (detail: `reports/hardware_profile.md`):
- Format panjang, semua toko, setelah buang hari sebelum penjualan pertama: 46,8 juta
  baris, 672 MB di RAM, 99,8 MB Parquet. Konversi per toko dengan numpy: puncak RAM 0,51 GB.
- Training 1 toko (CA_1, 4,5 juta baris, 27 fitur sementara) di GPU:
  `reg:tweedie` +514 MiB VRAM, puncak RAM 2,3 GB, 12 detik.
  4 kuantil dalam satu model: +790 MiB VRAM, tetapi **puncak RAM 10,5 GB**, 10,5 menit.
- **VRAM bukan hambatan; RAM host yang jadi hambatan**, lewat ukuran model.
  `item_id` sebagai fitur kategorikal membuat model ~40× lebih besar daripada label
  encoding, dan `xgb.train` (xgboost 3.x) menyalin model di akhir training (`reset()`).
- Fase 4 (fitur final, 36 fitur): tabel dasar per toko ±430 MB di RAM, ±70 MB Parquet;
  10 toko = 687 MB di disk. Training 1 toko: tweedie 4,3 GB RAM (label) / 4,6 GB
  (kategorikal); 4 kuantil dengan label encoding **5,0 GB** (turun dari 10,5 GB).
  Detail: `reports/hardware_profile.md` §3b.
- Fase 5: puncak RAM satu model terjadi saat `QuantileDMatrix` dibangun, bukan saat
  training. Melepas tabel fold sebelum konversi menurunkan puncak 4,87 → 4,43 GB (CA_3),
  dengan prediksi identik. Pembatas yang sebenarnya adalah **total RAM laptop**: aplikasi
  lain (VS Code, browser, Discord, dll.) bisa memakai ±7–11 GB dari 15,26 GB (terukur 24 Okt 2026). Eksperimen
  kuantil pertama dihentikan sistem karena itu. Tutup aplikasi besar saat training panjang.

**Aturan:**
1. **Granularitas default: satu model per toko** (`store_id`), dilatih di GPU dengan
   `device="cuda"`, `tree_method="hist"`. Perbandingan "global vs per toko" di Fase 5
   hanya boleh memakai model global yang dilatih dengan **sampel** data atau di CPU,
   dan batasan ini dicatat di laporan.
2. **Pipeline fitur berjalan per toko.** Fitur yang tidak bergantung cutoff (lag ≥ 28,
   rolling dari lag 28, kalender, harga) disimpan sekali ke
   `data/processed/base/store_id=<id>/`; fitur yang bergantung cutoff (target encoding,
   harga relatif rata-rata item) dihitung per fold saat training. Keamanannya dibuktikan
   tes anti-leakage di beberapa cutoff (`tests/test_features.py`). Jangan pernah memuat
   seluruh tabel fitur semua toko ke satu DataFrame.
3. **Kuantil:** coba dulu satu model multi-kuantil per toko. Kalau kena OOM di GPU,
   pindah ke **satu model per kuantil**, lalu catat pilihannya.
4. **Fallback OOM:** kalau training GPU gagal karena memori, (a) turunkan `max_bin`
   (mis. 256 → 128), (b) latih per toko × kategori, (c) pindah ke CPU
   (`device="cpu"`). Catat langkah yang dipakai di laporan run. Jangan diam-diam
   mengurangi data.
5. **Ukur, jangan menebak.** Di Fase 1, latih satu toko (konfigurasi default), lalu
   catat ke `reports/hardware_profile.md`:
   - puncak RAM (pakai `psutil`; `/usr/bin/time -v` tidak ada di Windows),
   - puncak VRAM (pantau `nvidia-smi --query-gpu=memory.used --format=csv -l 1`),
   - waktu training dan jumlah iterasi.
   Revisi estimasi di atas memakai angka ini.
6. **Iterasi cepat:** eksperimen (fitur, parameter, ablation) dilakukan di **1–2 toko**
   dulu. Semua 10 toko × semua fold hanya dijalankan untuk konfigurasi final.
7. **SHAP** dihitung pada sampel (mis. 100–200 ribu baris per toko), bukan seluruh data.
8. Laptop bisa *thermal throttling*. Jangan menyimpulkan perbedaan kecepatan antar
   eksperimen dari satu run.

---

## 5. Fase pengerjaan

### Fase 0 — Setup & verifikasi data
- Unduh M5, cek kolom & rentang `d_*` vs `calendar.csv`, catat ke `SOURCES.md`.
- Cek total baris setelah melt (perkiraan ±59 juta baris untuk 30.490 × 1.941),
  dan **laporkan angka sebenarnya**.

### Fase 1 — Pemrosesan efisien memori
- Downcast: unit sales → `int16`, harga → `float32`, ID → `category`.
- Simpan format panjang ke **Parquet**, dipartisi per `store_id`.
- Buang baris **sebelum penjualan pertama** tiap item-toko (produk belum dijual,
  bukan permintaan nol). Cek juga bahwa harganya memang tidak ada di periode itu.
- Proses per toko (wajib, lihat §4.1). Catat puncak RAM tiap langkah.
- Untuk training XGBoost: gunakan `xgboost.QuantileDMatrix` (bukan `DMatrix` biasa)
  dengan `tree_method="hist"` untuk menekan memori. Fitur dalam `float32`.
  Kalau tetap tidak muat, pertimbangkan mode *external memory* XGBoost (cek dokumentasi
  versi yang terpasang) atau latih per toko.

### Fase 2 — EDA terarah (bukan galeri grafik)
Jawab pertanyaan spesifik:
- Berapa proporsi hari dengan penjualan nol per kategori?
- Seberapa kuat pola mingguan dan efek hari SNAP per negara bagian?
- Hari apa toko tutup? (Cek apakah 25 Desember bernilai nol di semua toko. **Verifikasi
  dari data**, jangan berasumsi.) Tangani hari tutup secara eksplisit.
- Bagaimana bentuk penjualan di sekitar **Easter** (hari libur bergerak)? Ini persiapan P4.

### Fase 3 — Desain validasi (P1)
- Horizon: 28 hari. Gunakan **minimal 4 fold rolling origin** dengan cutoff mundur
  28 hari dari d_1913, contoh: d_1801, d_1829, d_1857, d_1885 (setiap fold memprediksi
  28 hari setelah cutoff-nya; fold terakhir memprediksi d_1886–d_1913). Fold tambahan
  yang mencakup akhir tahun disarankan.
- **Fold khusus** (dilaporkan terpisah, tidak dirata-rata ke skor utama):
  `easter_2014` (cutoff d_1164), `easter_2015` (d_1514), `year_end_2015` (d_1773).
  Detail & aturan: `reports/validation_design.md`, `config.yaml → validation`.
- Early stopping: 28 hari terakhir **sebelum** cutoff (di dalam jendela train).
- **Holdout final:** d_1914–d_1941 (cutoff d_1913). Hanya disentuh sekali di akhir.
  Tidak boleh ada fold CV yang horizon-nya masuk ke rentang ini.
- Semua fitur dihitung hanya dari data ≤ cutoff fold tersebut. Setiap fungsi fitur
  menerima `cutoff` dan wajib punya unit test anti-leakage.
- Laporkan rata-rata **dan sebaran** metrik antar fold. Kalau satu fold jauh berbeda,
  jelaskan penyebabnya.

### Fase 4 — Fitur
- **Lag & rolling** (aman untuk horizon 28 hari secara langsung): lag 28, 35, 42, 49, 56,
  364; rolling mean/std 7/14/28/56 pada lag 28. Kalau memakai lag < 28, pipeline harus
  rekursif dan didokumentasikan.
- **Harga:** harga relatif terhadap rata-rata item, perubahan harga minggu ke minggu,
  harga relatif terhadap item lain di departemen yang sama, indikator diskon.
- **Kalender:** hari dalam minggu, hari dalam bulan, event (tipe & nama), SNAP sesuai
  negara bagian toko.
- **Hari libur bergerak (P4):** untuk Easter, buat `days_to_event` dan `days_since_event`
  (dipotong pada ±N hari). Uji ablation **dengan vs tanpa** fitur ini di jendela sekitar
  Easter. Hasilnya menentukan apakah desain yang sama layak dipakai untuk Idulfitri.
  Easter 2016 (d_1885) tepat di batas fold CV 3/CV 4, jadi ablation butuh fold tambahan
  di sekitar Easter tahun sebelumnya (mis. 2015-04-05 = d_1528). Baca `event_name_1`
  **dan** `event_name_2` (Easter 2011 ada di kolom kedua). Lihat `data/SOURCES.md`.
- **Hierarki:** target encoding level item/dept/store **dihitung per fold** dari data train saja.
- **Stok habis (ditambahkan setelah EDA P5):** `zero_run_lag28` = panjang deretan nol per
  28 hari lalu. 62% item laris pernah tidak terjual ≥ 28 hari beruntun, indikasi kuat
  censored demand (`reports/eda.md` §P5).

### Fase 5 — Model & baseline
**Baseline wajib** (model harus dibandingkan secara jujur dengan semuanya):
1. Seasonal naive (nilai 7 hari lalu)
2. Moving average 28 hari
3. Croston / TSB untuk deret intermiten (`statsforecast`)
4. ETS untuk deret yang tidak intermiten (`statsforecast`), boleh di level agregat

**Model utama:** XGBoost, satu model per toko (lihat §4.1)
- **Point forecast:** objective `reg:tweedie` (cocok untuk data hitungan dengan banyak nol).
  Catatan: `count:tweedie` tidak ada di XGBoost (dicek di xgboost 3.4.1).
  Tuning `tweedie_variance_power` di rentang (1, 2), misalnya 1.1–1.5.
  Bandingkan dengan `count:poisson` dan `reg:squarederror`.
- **Kuantil (untuk P3):** objective `reg:quantileerror` dengan `quantile_alpha`
  (mis. `[0.1, 0.5, 0.8, 0.9]`). Di XGBoost ≥ 2.0 beberapa kuantil bisa dilatih dalam
  satu model. **Verifikasi** perilaku ini di versi yang terpasang. Cek juga
  *quantile crossing* (P90 < P50), dan kalau terjadi, urutkan ulang prediksi lalu catat
  seberapa sering itu terjadi.
- **Fitur kategorikal** (`item_id`, `dept_id`, `store_id`, dll.): pakai dtype `category`
  pandas + `enable_categorical=True` dengan `tree_method="hist"`. Bandingkan dengan
  label encoding biasa. Kardinalitas `item_id` tinggi (3.049), jadi pantau overfit,
  waktu training, **ukuran model dan puncak RAM** (Fase 1: kategorikal ~40× lebih besar,
  model 4 kuantil 1.000 iterasi = 1,6 GB).
- Model kuantil di `eta=0.05` tidak berhenti sebelum 3.000 iterasi (Fase 1). Tentukan
  `eta`/batas iterasi khusus kuantil supaya 10 toko × 4 fold tetap masuk akal waktunya.
- **Parameter awal yang di-tuning:** `max_depth`, `learning_rate` (eta), `min_child_weight`,
  `subsample`, `colsample_bytree`, `reg_lambda`, `max_bin`. Gunakan **early stopping**
  pada set validasi internal **di dalam** periode train fold (bukan pada horizon
  yang dievaluasi).
- **Hardware:** GPU default (`device="cuda"`, RTX 3050 4 GB), dengan fallback CPU sesuai
  §4.1. Catat device di laporan run (hasil CPU vs GPU bisa sedikit berbeda).
- Granularitas default **per toko** karena batasan VRAM (lihat §4.1). Per toko×kategori
  boleh diuji. Model global hanya dengan sampel atau CPU.
- Seed tetap (`random_state`). Simpan config, versi `xgboost`, `best_iteration`, dan
  metrik setiap run ke `reports/runs/<timestamp>/`.
- **Uji censored demand:** bandingkan training biasa vs training yang mengeluarkan baris di
  dalam deretan nol panjang item laris (seperti hari tutup). Evaluasi tetap pada penjualan
  aktual. Ambang deretan dan definisi "item laris" ditetapkan sebelum melihat hasil.

**Metrik:**
- WRMSSE (metrik resmi M5). Implementasikan sesuai panduan kompetisi M5 dan **verifikasi**
  terhadap referensi resmi atau implementasi publik yang sudah teruji sebelum dipercaya.
- Tambahan: RMSSE per deret, MAE, bias (mean error), pinball loss untuk kuantil.
- **Jangan pakai MAPE** (pembagian dengan nol).

### Fase 6 — Analisis error per segmen (P2)
- Klasifikasikan setiap deret dengan skema **Syntetos–Boylan** (ADI & CV²; ambang
  ADI = 1,32, CV² = 0,49): *smooth, erratic, intermittent, lumpy*. Hitung dari data train.
- Laporkan metrik model vs baseline **per segmen**, per kategori, per toko,
  per hari dalam minggu, dan **event vs non-event**.
- Cari jawaban jujur untuk: "Di segmen mana XGBoost *tidak* lebih baik dari Croston/naive?"
- SHAP: kepentingan fitur global **dan per segmen**. Gunakan TreeSHAP bawaan XGBoost
  (`booster.predict(dmatrix, pred_contribs=True)`) atau `shap.TreeExplainer`. Hitung pada
  **sampel** baris (mis. 100–500 ribu) agar tidak kehabisan memori. Jangan bandingkan
  dengan `feature_importances_` tipe *weight*, karena bias ke fitur dengan banyak split. Fitur harga dibaca sebagai
  asosiasi, **bukan** elastisitas kausal. Tulis peringatan ini di laporan.
- Studi kasus: 3–5 deret dengan error terbesar beserta penjelasannya (stockout?
  perubahan harga? produk baru?).

### Fase 7 — Framing bisnis: simulasi stok (P3)
- Simulasikan kebijakan *order-up-to* sederhana per item-toko pada periode holdout:
  stok target = forecast kuantil (P50 / P80 / P90) selama lead time.
- Parameter (`config.yaml`): lead time, frekuensi pemesanan, biaya kehabisan stok,
  biaya simpan. **M5 tidak memuat data biaya**, jadi nilai ini adalah **asumsi**:
  - tulis asumsinya secara eksplisit,
  - lakukan **analisis sensitivitas** pada beberapa rasio biaya.
- Output: tingkat layanan (fill rate), unit kehabisan stok, unit kelebihan stok,
  dan total biaya per kuantil, per segmen. Bandingkan model vs baseline.
- Keterbatasan wajib disebut: penjualan historis ≠ permintaan asli (*censored demand*
  saat stok habis). Pakai model yang dilatih tanpa deretan nol panjang (uji Fase 5),
  karena yang disimulasikan adalah permintaan.

### Fase 8 — Transfer ke Indonesia (P5)
**Langkah 1 — Audit kelayakan (wajib sebelum modeling):**
- Periode data, jumlah produk, jumlah transaksi per hari, persentase hari nol.
- Apakah periodenya mencakup **≥ 2 kali Ramadan/Idulfitri?**
- Tulis hasil audit ke `reports/id_data_audit.md`.

**Langkah 2 — Keputusan berdasarkan audit:**
- **Cukup** (≥ 2 tahun, cukup banyak produk aktif): jalankan pipeline yang sama
  (agregasi harian/mingguan per produk atau kategori), dengan fitur
  `days_to_idulfitri`, `is_ramadan`, `ramadan_day`, dan libur nasional.
  Lakukan ablation dengan vs tanpa fitur Ramadan.
- **Tidak cukup:** jangan memaksakan forecasting tingkat SKU. Ganti dengan
  **studi efek Ramadan/Idulfitri** (event study: penjualan relatif di jendela
  sebelum/selama/sesudah) dan forecasting di level agregat. Laporkan keterbatasannya
  dengan jujur.

**Langkah 3 — Bandingkan temuan:** apakah pola "hari libur bergerak" Easter di M5
dan Idulfitri di data Indonesia ditangkap dengan cara yang sama oleh fitur
`days_to_event`? Apa bedanya (durasi efek, kategori yang terdampak)?

### Fase 9 — Laporan
README berisi: pertanyaan, data, desain validasi, hasil vs baseline (dengan sebaran
antar fold), temuan per segmen, hasil simulasi stok, hasil transfer Indonesia,
**apa yang tidak berhasil**, dan keterbatasan.

---

## 6. Aturan kerja

1. **Data dulu, model belakangan.** Jangan menulis kode training sebelum data asli
   termuat, di-downcast, dan dicek (jumlah baris, rentang tanggal, persentase nol).
2. **Tidak ada data palsu.** Tes memakai potongan kecil data asli atau fixture yang
   jelas ditandai di `tests/fixtures/`.
3. **Anti-leakage:** setiap fitur bergantung pada `cutoff`, dengan unit test.
   Target encoding dan klasifikasi segmen dihitung dari data train per fold.
4. **Jangan mengarang** nama kolom, tanggal libur, atau angka hasil. Cek dari file atau
   jalankan kodenya, lalu tampilkan outputnya.
5. Sebelum menyatakan "selesai/berhasil", **jalankan** perintah terkait dan tunjukkan hasilnya.
6. Kalau model kalah dari baseline di suatu segmen, **laporkan**. Jangan disembunyikan
   atau "diperbaiki" dengan memilih fold yang menguntungkan.
7. Perhatikan RAM: hindari `.copy()` besar yang tidak perlu, gunakan Parquet dan
   pemrosesan per partisi.
8. Jangan menyalin notebook Kaggle publik. Referensi boleh dipakai untuk ide, tapi
   sumbernya dicatat di `README.md`.

---

## 7. Risiko yang sudah diketahui

- **Jenuh:** M5 sangat populer. Nilai proyek ini datang dari P1–P5, bukan dari skor.
- **XGBoost vs referensi M5:** banyak solusi teratas M5 memakai LightGBM, jadi angka
  dari notebook publik tidak bisa dibandingkan satu-satu. Bandingkan dengan baseline
  sendiri di fold yang sama. XGBoost di CPU juga bisa lebih lambat untuk ±59 juta baris,
  jadi mulai dari subset (mis. 1 toko) untuk iterasi cepat.
- **Permintaan intermiten:** banyak deret didominasi nol. Pastikan tidak ada model yang
  "menang" hanya karena memprediksi nol.
- **Censored demand:** penjualan nol bisa berarti stok habis, bukan tidak ada permintaan.
- **Data lama & konteks AS:** event (Super Bowl, SNAP) tidak berlaku di Indonesia.
  Yang ditransfer adalah **metode**, bukan pola.
- **Data Indonesia kecil/tidak pasti:** hasil transfer bisa saja lemah. Itu tetap temuan
  yang sah selama dilaporkan dengan jujur.
- **Tanggal Idulfitri** bisa berbeda antara perhitungan dan keputusan pemerintah.
  Selalu pakai sumber resmi.

---

## 8. Milestone

- [x] F0: akun Kaggle + token, unduh M5, verifikasi skema → `SOURCES.md`
      (sisa: kutip lisensi/aturan data M5 dari halaman Rules Kaggle)
- [x] F1: pipeline load/downcast/melt → Parquet, laporan jumlah baris & memori,
      profil RAM/VRAM training 1 toko → `reports/hardware_profile.md`
      (ulang profil di Fase 4 dengan fitur final)
- [x] F2: EDA terarah (nol per kategori, SNAP, hari tutup, Easter) → `reports/eda.md`
      (25 Des = tutup tapi tidak persis nol; efek Easter ±3 hari; SNAP nyata di FOODS)
- [x] F3: definisi fold rolling origin + tes anti-leakage → `reports/validation_design.md`
      (4 fold CV + 3 fold khusus + holdout; `leaking_columns` untuk tiap fitur di Fase 4)
- [x] F4: fitur + ablation fitur hari libur bergerak (Easter) → `reports/easter_ablation.md`
      (37 fitur termasuk `zero_run_lag28`; Easter: tidak terbukti di level item, membantu
      di total harian FOODS; fitur tetap dipakai)
- [x] F5: baseline (naive, MA, Croston/TSB, ETS) + XGBoost point (`reg:tweedie`) & kuantil (`reg:quantileerror`)
      → `reports/phase5_models.md`, `reports/quantile_cap1000.md`, `reports/quantile_cap3000_comparison.md`
      (WRMSSE diverifikasi terhadap bobot resmi M5. Point: tweedie 1,1, label encoding, tanpa
      deretan nol panjang, tanpa refit; CV WRMSSE 0,582 ± 0,051 vs classical 0,916, tetapi
      keunggulan di level item hanya 2–3%. Kuantil: satu model 4 kuantil per toko, eta 0,2,
      batas 1.000 (uji hingga 3.000 tidak membantu); P80/P90 coverage 81,0%/89,4%, tanpa
      crossing. Sisa: tuning parameter pohon (depth, min_child_weight, dll.) belum dilakukan;
      baseline kuantil dan model kuantil untuk fold khusus belum dibuat)
- [x] F6: analisis error per segmen ADI/CV² + SHAP + studi kasus → `reports/error_analysis.md`
      (per item XGBoost hanya 1–3% lebih baik dari classical di semua segmen, tetapi saat
      dijumlahkan ±54% lebih baik di intermittent/lumpy. XGBoost tidak lebih baik untuk total
      segmen erratic (bias +9%), akhir pekan per item, dan fold Easter 2015 per item. SHAP:
      te_item_mean 42%, harga 3%, event 0,3%; model lambat menyesuaikan level, sehingga item
      yang tumbuh −7% s.d. −32% dan item yang turun +5% s.d. +21%. Eksperimen te_item_mean jendela
      terbaru ditunda sampai Fase 7. SHAP GPU tidak aditif untuk fitur kategorikal, jadi
      dihitung di CPU)
- [ ] F7: simulasi stok + analisis sensitivitas biaya → `reports/inventory_sim.md`
      (**selesai di fold CV; simulasi holdout dijalankan sekali di akhir proyek.** Periodic review
      R=7, L=2, lost sales, biaya = asumsi. XGBoost point + safety stock paling efisien di stok
      6–10 hari; di layanan ≥ 98% empiris paling efisien. Asumsi dasar: empiris termurah,
      XGBoost kuantil +8,4%; empiris termurah di 10 dari 12 asumsi biaya. Jumlah kuantil harian
      tidak terkalibrasi (P50 → 28%, P90 → 97%). Metode gabungan ditolak. Item tumbuh cepat:
      classical termurah)
- [ ] F8: audit data apotek Indonesia → transfer atau studi efek Ramadan
- [ ] F9: README hasil & keterbatasan
