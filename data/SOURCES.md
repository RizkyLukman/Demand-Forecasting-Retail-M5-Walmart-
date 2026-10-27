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
- Kandidat anomali (sampai minggu d_1913): 405 baris < 20% median item-toko dan 34 baris
  > 5× median. Dibiarkan; lihat `reports/eda.md` §P5.

**Deretan nol panjang (censored demand)**
- 62% item laris (laku ≥ 50% hari) pernah tidak terjual ≥ 28 hari berturut-turut;
  29,5% dari semua hari bernilai nol berada di dalam deretan seperti itu. Indikasi kuat
  stok habis. Lihat `reports/eda.md` §P5.

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

## 1b. Bobot referensi WRMSSE (verifikasi metrik, Fase 5)

| Kolom | Nilai |
|---|---|
| Asal | `datasets/m5.zip` di repo GitHub Nixtla/m5-forecasts: https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip |
| Tanggal akses | 2026-10-23 |
| Lisensi | **Belum diverifikasi.** Dipakai hanya sebagai pembanding di tes, tidak disebarkan ulang. |
| File yang disimpan | `data/raw/m5_reference/weights_validation.csv` (tidak di-track git) |
| Ukuran | 2.242.706 byte, 42.840 baris data + 1 header |
| SHA-256 | `c957ad6429e59ac148918ddca13656d71784957c38af19e3314cad5c931572ab` |
| Kolom | `Level_id, Agg_Level_1, Agg_Level_2, Dollar_Sales, weight` |
| Dipakai di | `tests/test_metrics.py::test_weights_match_official_m5` |

- Bobot ini dihitung dari omzet 28 hari d_1886–d_1913, jadi **tidak memakai data holdout**
  (d_1914–d_1941).
- Hasil verifikasi: bobot yang dihitung `src/retailfc/metrics.py` untuk cutoff d_1913 cocok
  dengan file ini di ke-42.840 deret. Selisih bobot maksimum 2,3e-9, selisih relatif omzet
  maksimum 3,8e-7 (harga disimpan `float32`).
- **Hanya file ini yang diambil dari zip.** Zip itu juga memuat `sales_test_validation.csv`
  (penjualan d_1914–d_1941, yaitu jawaban holdout), `sales_test_evaluation.csv`, dan
  `weights_evaluation.csv`. Karena itu zip dihapus setelah file ini diekstrak, supaya jawaban
  holdout tidak tersedia di luar jalur resmi (`sales_train_evaluation.csv`, yang kolom
  holdout-nya hanya dibaca sekali di akhir proyek).

---

## 2. Retail sales dataset of a pharmacy in Indonesia

| Kolom | Nilai |
|---|---|
| Sitasi | Gustriansyah, R. (2022). *Retail sales dataset of a pharmacy in Indonesia*. Mendeley Data, V1. DOI `10.17632/2ym7v78wtd.1` |
| URL | https://data.mendeley.com/datasets/2ym7v78wtd/1 |
| Lisensi | CC BY 4.0 (http://creativecommons.org/licenses/by/4.0), menurut metadata API Mendeley |
| Cara unduh | API publik Mendeley: `https://data.mendeley.com/public-api/datasets/2ym7v78wtd/files?folder_id=root&version=1` → `download_url` |
| Tanggal akses | 2026-10-28 |
| Tanggal terbit | 2022-02-14 (versi 1, satu-satunya versi) |
| Lokasi | `data/raw/id_pharmacy/` |

| File | Byte | SHA-256 |
|---|---:|---|
| `sales.zip` | 8.560.326 | `5b77caa777993e0b01b9dd714cfaefe3aecd9b2b4b9138d550fec43d4258ca94` (sama dengan `sha256_hash` di metadata Mendeley) |
| `sales.sql` (isi zip) | 70.610.765 | `89e6ddb34c06c6737e575bab667e2c96506883d961abff439c018df0035f3fa0` |

- Isi: dump MariaDB 10.4.6 (HeidiSQL 11.3) dengan tabel `det_sales`, `ms_product`,
  `ms_sales`, `transaction`. Deskripsi Mendeley menyebut `ms_products`; nama di dump
  `ms_product`.
- Periode: 1 Jan–31 Des 2015. Header nota Jan–Mar sebagian besar hilang. Detail:
  `reports/id_data_audit.md`.
- Kolom `KD_CUST` dan `KD_DOKTER` berisi kode pasien/dokter; hanya dipakai untuk agregat.
