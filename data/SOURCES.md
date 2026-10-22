# Sumber data

Semua angka di bawah berasal dari `python -m retailfc.verify_m5 --config config.yaml`
(output lengkap: `reports/m5_verification.json`). File mentah ada di `data/raw/` dan
tidak di-track git.

---

## 1. M5 Forecasting – Accuracy (Walmart)

| Kolom | Nilai |
|---|---|
| URL | https://www.kaggle.com/competitions/m5-forecasting-accuracy/data |
| Cara unduh | `kaggle competitions download -c m5-forecasting-accuracy` (Kaggle CLI 2.2.4) |
| Tanggal akses | 2026-10-22 |
| Lisensi / ketentuan | **Belum diverifikasi.** Diatur oleh aturan kompetisi yang disetujui di Kaggle (Late Submission). Bagian "Competition Data" di https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules harus dibaca dalam keadaan login lalu dikutip di sini. |
| Lokasi | `data/raw/m5/` |

### File dan checksum

| File | Byte | SHA-256 |
|---|---:|---|
| `m5-forecasting-accuracy.zip` | 48.009.163 | `0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5` |
| `calendar.csv` | 103.469 | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `sales_train_evaluation.csv` | 121.736.518 | `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| `sales_train_validation.csv` | 120.007.726 | `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724` |
| `sample_submission.csv` | 5.228.786 | `eb7597afbec367475ef25dc6dbbc009b8faa47fa2107252c72a029e3cdf72343` |
| `sell_prices.csv` | 203.395.785 | `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |

### Cek skema (dokumentasi vs file asli)

Ketiga skema di PROJECT_PLAN.md §2.1 **sama persis dengan file** (kolom sama, urutan
sama, tidak ada yang hilang atau berlebih):

- `calendar.csv`: `date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI`
- `sell_prices.csv`: `store_id, item_id, wm_yr_wk, sell_price`
- `sales_train_evaluation.csv`: `id, item_id, dept_id, cat_id, store_id, state_id, d_1 … d_1941`

### Fakta yang sudah diverifikasi

**Kalender**
- 1.969 baris, `d_1` … `d_1969`, 2011-01-29 → 2016-06-19, tanpa celah tanggal maupun `d`.
- Kalender 28 hari lebih panjang dari data penjualan (`d_1942` … `d_1969` = jendela
  evaluasi privat kompetisi, tanpa label).
- 162 hari event; 5 hari punya event kedua (`event_name_2`).
- Tipe event: Cultural, National, Religious, Sporting.
- **Easter** (dari file): d_86 2011-04-24, d_436 2012-04-08, d_793 2013-03-31,
  d_1178 2014-04-20, d_1528 2015-04-05, d_1885 2016-03-27.
  Di 2011 Easter ada di `event_name_2` (OrthodoxEaster di `event_name_1`), jadi fitur
  event harus membaca **kedua** kolom event.

**Penjualan (`sales_train_evaluation.csv`)**
- 30.490 deret (`id` unik), 3.049 item, 10 toko (CA_1–4, TX_1–3, WI_1–3),
  7 departemen, 3 kategori (FOODS, HOBBIES, HOUSEHOLD), 3 negara bagian.
- `d_1` … `d_1941` berurutan; semuanya ada di `calendar.csv`.
- Nilai 0 … 763 → muat di `int16`. Di memori sebagai `int16`: 115,6 MB.
- **Baris setelah melt: 59.181.090** (30.490 × 1.941). Estimasi di PROJECT_PLAN.md tepat.
- Proporsi nol, semua sel: 68,0%.
- Sel sebelum penjualan pertama tiap deret: 12.384.870. Setiap deret punya minimal satu
  penjualan. Setelah sel-sel itu dibuang (Fase 1), tersisa **46.796.220** baris dengan
  proporsi nol 59,5%.

**Harga vs penjualan pertama (dicek di Fase 1, `reports/phase1_io.json`)**
- Tidak ada item-toko yang terjual sebelum punya harga (0 deret), dan tidak ada baris
  tersimpan yang tanpa harga.
- Pada 30.346 dari 30.490 deret (99,5%), minggu harga pertama **sama dengan** minggu
  penjualan pertama. Harga dicatat per minggu, jadi 85.457 baris yang dibuang (hari-hari
  lebih awal di minggu yang sama) memang punya harga.
- **144 deret** punya harga 1–225 minggu (median 3,5) sebelum penjualan pertamanya.
  Minggu-minggu itu ikut dibuang oleh aturan "sebelum penjualan pertama", meskipun
  produknya sudah terdaftar. Bisa jadi permintaannya memang nol, atau ada stockout yang
  tidak tercatat. Dicatat sebagai pengecualian yang diketahui.

**Harga (`sell_prices.csv`)**
- 6.841.121 baris, 10 toko, 3.049 item, `wm_yr_wk` 11101 … 11621.
- Rentang harga 0,01 … 107,32, tanpa NaN, tanpa kunci (toko, item, minggu) ganda.

**File validation vs evaluation**
- `sales_train_validation.csv` berakhir di `d_1913`; urutan deretnya sama dengan file
  evaluation, dan hari `d_1` … `d_1913` **identik**. File evaluation mencakup seluruh isi
  file validation, jadi hanya file evaluation yang dipakai.

**Pemakaian sumber daya pengecekan:** puncak RAM 0,57 GB, 7,4 detik.

### Batas fold (tanggal dari `calendar.csv`)

| Fold | Cutoff | Horizon | Tanggal horizon |
|---|---|---|---|
| CV 1 | d_1801 | d_1802–d_1829 | 2016-01-04 → 2016-01-31 |
| CV 2 | d_1829 | d_1830–d_1857 | 2016-02-01 → 2016-02-28 |
| CV 3 | d_1857 | d_1858–d_1885 | 2016-02-29 → 2016-03-27 |
| CV 4 | d_1885 | d_1886–d_1913 | 2016-03-28 → 2016-04-24 |
| Holdout | d_1913 | d_1914–d_1941 | 2016-04-25 → 2016-05-22 |

Catatan untuk P4: Easter 2016 (d_1885) adalah **hari terakhir horizon CV 3**, dan CV 4
dimulai sehari sesudahnya. Tidak ada satu fold pun yang mencakup jendela sekitar Easter
2016. Ablation Easter butuh fold tambahan di sekitar Easter tahun sebelumnya (mis.
2015-04-05 = d_1528). Fold tambahan ini dibuat di Fase 3 (`reports/validation_design.md`).

---

## 2. Retail sales dataset of a pharmacy in Indonesia

Belum diunduh (Fase 8). DOI `10.17632/2ym7v78wtd.1`, CC BY 4.0.
