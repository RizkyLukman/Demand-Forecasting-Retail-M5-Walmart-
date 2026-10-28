# Fase 5d: Tuning Parameter Pohon XGBoost

*[← 05c Uji batas iterasi](05c_quantile_cap3000.md) · [Daftar laporan](README.md) · [06 Analisis error →](06_error_analysis.md)*

> **Status (29 Okt 2026):** selesai. **Tidak ada parameter yang diganti.** Konfigurasi final tetap
> `config.yaml → model.xgb_point`. Kode: `src/retailfc/xgb_tuning.py`. Angka lengkap:
> `reports/data/xgb_tuning.json` (2 toko) dan `reports/data/model_comparison_xgb_point_tuned.json`
> (konfirmasi 10 toko). Holdout tidak disentuh.

---

## Ringkasan

- Enam kelompok parameter diuji berurutan di 2 toko × 4 fold CV. Hanya satu yang lolos:
  `min_child_weight` 10 → 200 (WRMSSE 0,6903 → 0,6803).
- Saat dikonfirmasi di **10 toko × 4 fold**, perbaikannya menyusut menjadi **0,0030**
  (0,58194 → 0,57897), di bawah ambang noise 0,0071. Sesuai aturan yang ditetapkan sebelum
  run, kandidat ini **ditolak**.
- Arah perbaikannya konsisten (lebih baik di 3 dari 4 fold dan di 11 dari 12 level hierarki),
  tetapi besarnya hanya ±0,5%. Itu tidak sebanding dengan menjalankan ulang model kuantil, simulasi
  stok, analisis error, dan SHAP.
- Kesimpulannya: konfigurasi awal (depth 8, min_child_weight 10, eta 0,05, sampling 0,8/0,8)
  sudah berada di dekat titik optimum untuk set fitur ini. Ruang perbaikan berikutnya lebih
  mungkin ada di fitur (Fase 6: model lambat menyesuaikan level), bukan di parameter pohon.

## 1. Desain (ditetapkan sebelum run)

| Item | Nilai |
|---|---|
| Data | CA_3 dan WI_2 × cv1–cv4, sama dengan eksperimen Fase 5 |
| Titik awal | `model.xgb_point` (tweedie 1,1, label encoding, tanpa deretan nol panjang) |
| Noise | titik awal dengan seed 42 vs 43: 0,69025 vs 0,69730 → **tau = 0,00705** |
| Urutan | satu kelompok parameter per tahap, mulai dari yang terbaik di tahap sebelumnya |
| Aturan menerima | rata-rata lebih baik > tau **dan** lebih baik di ≥ 3 dari 4 fold |
| Konfirmasi | aturan yang sama, 10 toko × 4 fold, terhadap model final saat ini |

Titik awal menghasilkan WRMSSE 0,69025, **sama persis** dengan run Fase 5. Artinya training
deterministik dan hasilnya bisa dibandingkan langsung.

## 2. Hasil per tahap (2 toko, WRMSSE rata-rata 4 fold)

| Tahap | Kandidat | WRMSSE | Keputusan |
|---|---|---:|---|
| awal | depth 8, mcw 10, sampling 0,8/0,8, lambda 1, bin 256, eta 0,05 | 0,69025 | |
| T1 max_depth | 6 / 10 / 12 | 0,68712 / 0,70361 / 0,74792 | ditolak (6: selisih 0,003 < tau) |
| T2 min_child_weight | 1 / 50 / **200** | 0,69777 / 0,69890 / **0,68027** | **diterima** (−0,010, 3/4 fold) |
| T3 subsample/colsample | 0,6/0,8; 1,0/0,8; 0,8/0,5; 0,8/1,0 | 0,67456 / 0,68028 / 0,68943 / 0,68308 | ditolak (0,6/0,8: selisih 0,0057 < tau) |
| T4 reg_lambda | 5 / 20 / 100 | 0,69289 / 0,69198 / 0,68976 | ditolak (semua lebih buruk) |
| T5 max_bin | 128 / 512 | 0,69058 / 0,70929 | ditolak (keduanya lebih buruk) |
| T6 eta | 0,03 / 0,1 | 0,67966 / 0,68209 | ditolak (0,03: selisih 0,0006; 35% lebih lambat) |

Pola yang terlihat: semua perubahan yang menambah kapasitas model (pohon lebih dalam, bin lebih
halus) memperburuk hasil. Perubahan yang membuat model lebih "hati-hati" (syarat bukti per daun
lebih ketat, sampling baris lebih agresif) sedikit membantu. Rem tambahan lewat `reg_lambda`
tidak membantu setelah `min_child_weight` dinaikkan.

## 3. Konfirmasi 10 toko × 4 fold

| Model | WRMSSE (± sd) | cv1 | cv2 | cv3 | cv4 |
|---|---:|---:|---:|---:|---:|
| `xgb_point` (sekarang) | 0,58194 ± 0,051 | 0,5659 | 0,6379 | 0,6037 | 0,5203 |
| `xgb_point_tuned` (mcw 200) | 0,57897 ± 0,047 | 0,5645 | 0,6322 | 0,5979 | 0,5213 |
| selisih | **−0,0030** | −0,0014 | −0,0057 | −0,0058 | +0,0010 |

- Lebih baik di 3 dari 4 fold, tetapi rata-ratanya di bawah tau → **ditolak**.
- Per level hierarki, kandidat lebih baik di 11 dari 12 level (−0,1% s.d. −1,3%), dan lebih
  buruk di level total (+0,9%). MAE 1,0526 → 1,0517; RMSE 2,2338 → 2,2282.
- Perbaikan di 2 toko (−0,010) menyusut ±70% saat diuji di 10 toko. Ini contoh "pemenang
  terlihat lebih bagus dari kenyataannya": konfigurasi terbaik dari banyak kandidat di sampel
  kecil cenderung melebih-lebihkan perbaikannya. Karena itu langkah konfirmasi dibutuhkan.

**Catatan soal ambang:** tau diukur di 2 toko. Noise antar-seed di 10 toko kemungkinan lebih
kecil, karena errornya dirata-rata dari lebih banyak deret, sehingga perbaikan 0,003 bisa jadi
nyata. Namun aturan "sama dengan 2 toko" ditetapkan sebelum run, dan tidak saya ubah setelah
melihat hasilnya. Kalaupun nyata, perbaikan 0,5% tidak mengubah kesimpulan proyek.

## 4. Bug yang ditemukan

`train.py` tidak meneruskan `max_bin` ke matriks validasi (`QuantileDMatrix(..., ref=dtrain)`),
sehingga xgboost 3.4.1 menolak `max_bin` ≠ 256. Bug ini baru muncul di tahap T5 dan sudah
diperbaiki. Semua run sebelumnya memakai `max_bin` 256, jadi hasilnya tidak terpengaruh.

## 5. Biaya

19 konfigurasi × 8 model di 2 toko (70 menit), ditambah 40 model untuk konfirmasi (±20 menit).
Puncak RAM 4,5–4,9 GB per konfigurasi di 2 toko, dan 5,3 GB saat konfirmasi 10 toko. `max_depth` 12 membuat model ±8× lebih besar (30 MB vs
4 MB) dan ±75% lebih lambat.
