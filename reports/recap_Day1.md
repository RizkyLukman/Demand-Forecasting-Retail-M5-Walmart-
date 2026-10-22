# Recap Hari 1 (22–23 Oktober 2026)

Ringkasan semua yang sudah dikerjakan di proyek **Demand Forecasting Retail (M5 Walmart →
Indonesia) dengan XGBoost**: tujuan, apa yang dilakukan di tiap fase, hasil analisis,
keputusan, kendala, dan apa yang masih terbuka.

Laporan detail tiap fase tetap ada di file masing-masing (ditautkan di bawah). Recap ini
memuat inti analisisnya sehingga bisa dibaca sendiri.

---

## Daftar isi

1. [Gambaran besar](#1-gambaran-besar)
2. [Status per fase](#2-status-per-fase)
3. [Fase 0 — Setup & verifikasi data](#3-fase-0--setup--verifikasi-data)
4. [Fase 1 — Pemrosesan efisien memori & profil hardware](#4-fase-1--pemrosesan-efisien-memori--profil-hardware)
5. [Fase 2 — EDA terarah](#5-fase-2--eda-terarah)
6. [Fase 3 — Desain validasi](#6-fase-3--desain-validasi)
7. [Fase 4 — Fitur & ablation Easter](#7-fase-4--fitur--ablation-easter)
8. [Preprocessing & kualitas data](#8-preprocessing--kualitas-data)
9. [Kendala yang ditemui & cara mengatasinya](#9-kendala-yang-ditemui--cara-mengatasinya)
10. [Keputusan yang sudah diambil](#10-keputusan-yang-sudah-diambil)
11. [Hal yang masih terbuka](#11-hal-yang-masih-terbuka)
12. [Riwayat commit & isi repo](#12-riwayat-commit--isi-repo)
13. [Langkah berikutnya](#13-langkah-berikutnya)

---

## 1. Gambaran besar

### Pertanyaan proyek
1. Seberapa akurat XGBoost meramalkan penjualan harian **per produk per toko** untuk
   **28 hari ke depan**?
2. Di segmen produk mana model gagal?
3. Apa dampaknya terhadap keputusan stok (kehabisan vs kelebihan stok)?
4. Apakah pipeline yang sama bisa dipindahkan ke data retail Indonesia, dengan efek
   kalender hijriah (Ramadan/Idulfitri)?

### Kenapa tidak berhenti di "XGBoost + skor"
Data M5 sudah dibahas di ribuan notebook publik. Nilai proyek ini ada di lima pembeda:

| # | Pembeda | Fase | Status |
|---|---|---|---|
| P1 | Backtesting jujur: beberapa fold rolling origin | 3 | ✅ desain selesai |
| P2 | Analisis error per segmen | 6 | belum |
| P3 | Forecast kuantil → simulasi kebijakan stok | 7 | belum |
| P4 | Fitur hari libur bergerak, divalidasi di Easter dulu | 4 & 8 | ✅ bagian Easter selesai |
| P5 | Transfer ke data apotek Indonesia | 8 | belum |

### Aturan kerja yang dipegang
- **Data dulu, model belakangan.** Tidak ada kode training sebelum data dicek.
- **Tidak ada data palsu** dan tidak ada angka atau tanggal yang dikarang; semua dicek dari
  file atau dari output kode.
- **Anti-leakage:** setiap fitur hanya boleh memakai data ≤ cutoff, dibuktikan dengan tes.
- **Holdout (d_1914–d_1941) belum disentuh sama sekali**, dan baru dipakai sekali di akhir.
- Hasil yang tidak sesuai harapan **tetap dilaporkan**.
- Commit atas nama pemilik repo; laporan, teks grafik, komentar, dan docstring dalam
  bahasa Indonesia.

### Lingkungan
| Item | Nilai |
|---|---|
| Laptop | RAM 16 GB, GPU RTX 3050 Laptop 4 GB, CPU AMD 16 core logis |
| Python | 3.12.5, venv di `C:\Users\ASUS\.venvs\retailfc` (di luar folder proyek, lihat §9) |
| Library utama | xgboost 3.4.1 (CUDA), pandas 2.3.3, numpy 2.5.4, statsforecast 2.1.1, shap 0.54.1 |
| Repo | https://github.com/RizkyLukman/Demand-Forecasting-Retail-M5-Walmart- (branch `main`) |

---

## 2. Status per fase

| Fase | Isi | Status | Laporan detail |
|---|---|---|---|
| 0 | Setup repo, unduh & verifikasi data M5 | ✅ (sisa: kutip lisensi) | `data/SOURCES.md` |
| 1 | Format panjang hemat memori + profil hardware | ✅ | `reports/hardware_profile.md` |
| 2 | EDA terarah (4 pertanyaan + kualitas data) | ✅ | `reports/eda.md` |
| 3 | Desain validasi (fold + anti-leakage) | ✅ | `reports/validation_design.md` |
| 4 | Fitur + ablation Easter + profil ulang | ✅ (belum di-commit) | `reports/easter_ablation.md` |
| 5 | Baseline + training XGBoost utama | belum | – |
| 6 | Analisis error per segmen + SHAP | belum | – |
| 7 | Simulasi stok | belum | – |
| 8 | Transfer ke data Indonesia | belum | – |
| 9 | README hasil & keterbatasan | belum | – |

Tes otomatis: **61 tes, semua lolos** (`pytest -q`).

---

## 3. Fase 0 — Setup & verifikasi data

### Tujuan
Menyiapkan repo dan memastikan data M5 yang dipakai **benar-benar sesuai** dengan
dokumentasi, sebelum dipakai untuk apa pun.

### Yang dilakukan
- Merapikan dokumen rencana: nomor fase yang tidak konsisten, perintah Windows, default
  model per toko di GPU, dan **fold CV terakhir yang bocor ke holdout**. Cutoff awalnya
  d_1913, yang horizonnya sama persis dengan holdout; diganti menjadi d_1801/1829/1857/1885.
- Membuat struktur repo, `pyproject.toml`, `config.yaml`, `.gitignore`.
- Mengunduh M5 lewat Kaggle CLI, lalu menjalankan `python -m retailfc.verify_m5`.

### Hasil verifikasi
| Pengecekan | Hasil |
|---|---|
| Skema 3 file vs dokumentasi | **Sama persis** (kolom dan urutannya) |
| Jumlah deret | 30.490 (3.049 item × 10 toko), 7 departemen, 3 kategori, 3 negara bagian |
| Rentang hari | d_1–d_1941 = 2011-01-29 → 2016-05-22, tanpa celah |
| Kalender | d_1–d_1969 (28 hari lebih panjang = periode evaluasi privat kompetisi) |
| Baris setelah melt | **59.181.090** (estimasi rencana tepat) |
| Proporsi nol (semua sel) | 68,0% |
| Sel sebelum penjualan pertama | 12.384.870 → sisa 46.796.220 baris setelah dibuang |
| Nilai penjualan | 0–763 (muat di `int16`) |
| Harga | 6.841.121 baris, 0,01–107,32, tanpa NaN dan tanpa duplikat |
| File validation vs evaluation | Hari d_1–d_1913 **identik** → cukup pakai file evaluation |
| Checksum SHA-256 semua file | Dicatat di `data/SOURCES.md` |

### Temuan penting
- **Easter 2011 tercatat di kolom `event_name_2`**, bukan kolom pertama. Fitur event
  harus membaca kedua kolom.
- **Easter 2016 (d_1885) tepat di batas fold CV 3 dan CV 4**, sehingga tidak ada satu
  fold pun yang mencakup jendela sekitar Easter 2016. Ini memicu dibuatnya fold khusus
  Easter di Fase 3.

---

## 4. Fase 1 — Pemrosesan efisien memori & profil hardware

### Tujuan
Mengubah data ke bentuk yang siap dipakai (satu baris per item per hari) **tanpa
kehabisan RAM**, lalu **mengukur** (bukan menebak) kebutuhan RAM/VRAM untuk training.

### Yang dilakukan
- `python -m retailfc.io`: format panjang per toko langsung dengan numpy (tanpa
  `pandas.melt`), tipe data diperkecil (`int16`, `float32`, `category`), hari sebelum
  penjualan pertama dibuang, lalu disimpan ke Parquet per toko.
- Mengecek apakah harga memang belum ada sebelum penjualan pertama.
- `python -m retailfc.hardware_profile`: melatih 1 toko (CA_1) dan mengukur RAM, VRAM,
  serta waktu.

### Hasil konversi data
| Item | Hasil |
|---|---|
| Baris tersimpan | 46.796.220 (sama dengan perhitungan Fase 0) |
| Total unit sebelum vs sesudah | 66.927.173 = 66.927.173 (tidak ada yang hilang) |
| Baris tersimpan tanpa harga | 0 |
| Puncak RAM | **0,51 GB** (estimasi awal 1–2 GB terlalu tinggi) |
| Ukuran | 672 MB di RAM, 99,8 MB Parquet |

### Analisis harga vs penjualan pertama
- 99,5% deret (30.346 dari 30.490): minggu harga pertama = minggu penjualan pertama. Harga
  tercatat per minggu, jadi wajar bila harga muncul beberapa hari sebelum penjualan
  pertama di minggu yang sama.
- **144 deret** punya harga 1–225 minggu sebelum penjualan pertamanya (produk sudah
  terdaftar tapi belum laku, atau stok kosong yang tidak tercatat). Dicatat sebagai
  pengecualian.

### Profil hardware (fitur sementara, 27 fitur)
| Model | Waktu | Puncak RAM | VRAM |
|---|---|---:|---:|
| `reg:tweedie` (point forecast) | 11,8 s | 2,3 GB | +514 MiB |
| 4 kuantil dalam satu model | 632 s | **10,5 GB** | +790 MiB |

### Investigasi lonjakan RAM 10,5 GB
Masalah ini diselidiki dengan pengukuran bertahap:
1. Selama training, RAM hanya naik pelan (2,30 → 2,47 GB dalam 1.000 iterasi).
2. Lonjakan terjadi **di akhir `xgb.train`**: XGBoost 3.x memanggil `reset()` yang
   menyalin seluruh model.
3. Ukuran model membengkak karena **`item_id` (3.049 kategori) dipakai sebagai fitur
   kategorikal**:

| 100 iterasi, 4 kuantil | Ukuran model |
|---|---:|
| `item_id` kategorikal | 283,9 MB |
| `item_id` label encoding | **6,9 MB** (41× lebih kecil) |

**Kesimpulan Fase 1:** VRAM bukan hambatan; **RAM host** yang jadi hambatan, lewat ukuran
model. Ini menjadi masukan langsung untuk keputusan encoding di Fase 5.

---

## 5. Fase 2 — EDA terarah

### Tujuan
Menjawab pertanyaan spesifik yang menentukan desain fitur, **bukan** membuat galeri
grafik. Hanya data d_1–d_1913 yang dipakai (holdout tidak disentuh).

### P1. Berapa proporsi hari tanpa penjualan?
| Kategori | Proporsi nol |
|---|---:|
| FOODS | 51,1% |
| HOUSEHOLD | 64,6% |
| HOBBIES | 71,5% |
| **Semua** | **59,6%** |

→ Model bisa terlihat bagus hanya dengan menebak nol. Evaluasi wajib dilakukan per segmen
dan dengan mengecek bias.

### P2a. Seberapa kuat pola mingguan?
Akhir pekan terjual **37–48% lebih banyak** daripada Selasa–Kamis. HOBBIES memuncak di hari
Sabtu lalu turun di hari Minggu; FOODS dan HOUSEHOLD tetap tinggi di hari Minggu.
→ Hari dalam minggu wajib jadi fitur; lag dibuat kelipatan 7.

### P2b. Apakah efek SNAP (kupon makanan) nyata?
SNAP hanya bisa dipakai membeli makanan, jadi HOBBIES dan HOUSEHOLD dipakai sebagai
**pembanding (placebo)** untuk memisahkan efek SNAP dari efek awal bulan seperti gajian.

| Negara bagian | FOODS | HOUSEHOLD | HOBBIES |
|---|---:|---:|---:|
| CA | +10,2% | +3,8% | +2,7% |
| TX | +14,9% | +2,6% | +1,9% |
| WI | **+28,1%** | +3,0% | +1,3% |

→ Efek SNAP pada FOODS nyata, paling kuat di WI. Fitur `snap` dipakai sesuai negara bagian
toko. (Ini asosiasi, bukan bukti kausal.)

### P3. Hari apa toko tutup?
**25 Desember praktis tutup, tetapi tidak persis nol**: seluruh jaringan hanya menjual
11–20 unit. Pengecekan "total = 0" saja akan melewatkannya. Tidak ada hari lain yang
penjualannya di bawah 10% normal.
→ Natal diperlakukan sebagai hari tutup: dikeluarkan dari training dan statistik rolling,
forecast-nya diisi 0.

### P4. Seperti apa penjualan di sekitar Easter?
| Hari dari Easter | −2 (Jum) | −1 (Sab) | 0 (Min) | +2 |
|---|---:|---:|---:|---:|
| FOODS | **+16%** | **+15%** | −9% | −1% |
| HOBBIES | −1% | −2% | **−22%** | **+13%** |
| HOUSEHOLD | +9% | 0% | **−17%** | +4% |

→ Efeknya pendek (sekitar ±3 hari) dan tidak bisa ditangkap oleh penanda event di hari-H
saja. Ini dasar fitur `days_to/since_easter` di Fase 4.

### P5. Kualitas data (ditambahkan setelah pertanyaan soal preprocessing)
Lihat §8 di bawah. Ringkasnya: harga anomali sangat sedikit, tetapi **29,5% dari semua
hari bernilai nol kemungkinan besar adalah stok habis**, bukan tanpa pembeli.

Grafik: `reports/figures/eda_weekday.png`, `reports/figures/eda_easter.png`.

---

## 6. Fase 3 — Desain validasi

### Tujuan
Membangun cara evaluasi yang **jujur** (P1): model diuji di beberapa periode berbeda, tidak
pernah melihat masa depan, dan holdout benar-benar terpisah.

### Fold yang dibuat
| Fold | Peran | Horizon (28 hari) | Isi penting |
|---|---|---|---|
| cv1 | utama | 2016-01-04 → 01-31 | – |
| cv2 | utama | 2016-02-01 → 02-28 | Super Bowl, Valentine |
| cv3 | utama | 2016-02-29 → 03-27 | Easter 2016 di hari terakhir |
| cv4 | utama | 2016-03-28 → 04-24 | tanpa event |
| easter_2014 | khusus | 2014-04-07 → 05-04 | Easter 2014 |
| easter_2015 | khusus | 2015-03-23 → 04-19 | Easter 2015 |
| year_end_2015 | khusus | 2015-12-07 → 2016-01-03 | Natal + Tahun Baru |
| holdout | tes akhir | 2016-04-25 → 05-22 | dipakai sekali |

- Skor utama = rata-rata **dan sebaran** cv1–cv4 (berurutan, tanpa celah/tumpang tindih).
- Fold khusus dilaporkan terpisah, tidak dirata-rata ke skor utama.
- Early stopping memakai 28 hari terakhir **sebelum** cutoff, tidak pernah horizon.
- Kode menolak fold yang melanggar aturan (menyentuh holdout, tumpang tindih, riwayat
  kurang dari 420 hari).

### Alat anti-leakage
Cara kerjanya: semua penjualan setelah cutoff diacak, lalu fitur dihitung ulang. Kalau ada
fitur yang berubah, berarti fitur itu "mengintip masa depan". Alat ini sudah dibuktikan
bekerja pada data asli:

| Fitur uji | Seharusnya | Hasil |
|---|---|---|
| lag 28/35/364, rolling dari lag 28, rata-rata item dari data train, harga | aman | lolos |
| lag 7, rata-rata item dari semua data, rolling berpusat, lag 28 untuk horizon 29 | bocor | **tertangkap** |

---

## 7. Fase 4 — Fitur & ablation Easter

### Tujuan
Membuat fitur final yang aman dari leakage, lalu menguji apakah fitur hari libur bergerak
(Easter) benar-benar menambah akurasi (P4), sebagai uji coba sebelum dipakai untuk
Idulfitri.

### 37 fitur final
| Kelompok | Fitur |
|---|---|
| ID | `item_id`, `dept_id`, `cat_id` |
| Lag | lag 28, 35, 42, 49, 56, 364 |
| Rolling (dari lag 28) | mean & std 7/14/28/56 hari |
| Stok habis | `zero_run_lag28`: panjang deretan nol per 28 hari lalu (ditambahkan setelah §8) |
| Harga | harga, perubahan 7 hari, relatif terhadap departemen, relatif terhadap harga maks 56 hari, indikator diskon, relatif terhadap rata-rata item* |
| Kalender | hari dalam minggu, tanggal, bulan, tahun, `event_name_1/2`, `event_type_1`, `snap` |
| Hari libur bergerak | `days_to_easter`, `days_since_easter` (dibatasi 7 hari) |
| Target encoding* | rata-rata penjualan item, proporsi nol item, rata-rata departemen |

\* dihitung per fold dari data ≤ cutoff saja.

- **Desain penyimpanan:** fitur yang tidak bergantung pada cutoff disimpan sekali per
  toko (10 tabel, 709 MB); fitur yang bergantung pada cutoff dihitung saat training. Ini
  menggantikan rencana awal "1 tabel per fold × toko" (80 tabel).
- Semua fungsi fitur lolos tes anti-leakage di 5 cutoff berbeda, dan nilainya cocok
  dengan perhitungan langsung dari data.
- Hari Natal diberi nilai kosong sebelum lag/rolling dihitung.

### Ablation Easter (P4)
Setup ditetapkan **sebelum** melihat hasil: 2 toko (CA_1, TX_1) × 4 fold (easter_2014,
easter_2015, cv3, cv4) × 2 varian (dengan/tanpa fitur Easter) × 2 seed = 32 model.
(Dijalankan dengan 36 fitur, sebelum `zero_run_lag28` ditambahkan.)

**Kriteria:** fitur berguna jika error di jendela ±7 hari sekitar Easter turun konsisten
di semua fold dan toko, dan penurunannya lebih besar dari selisih antar seed.

**Hasil level item: kriteria TIDAK terpenuhi.**

| Metrik (dengan − tanpa) | Rata-rata | Membaik |
|---|---:|---:|
| RMSE jendela Easter | +0,0002 | 7/16 |
| MAE jendela Easter | +0,0020 | 9/16 |
| RMSE seluruh horizon | +0,0024 | 8/16 |
| *Noise antar seed* | *0,031* | – |

**Hasil level total harian: fitur membantu di FOODS sebelum Easter.**

| FOODS, prediksi/aktual | hari −2 | hari −1 |
|---|---:|---:|
| tanpa fitur Easter | 0,78 | 0,82 |
| dengan fitur Easter | **0,87** | **0,88** |

**Analisis:** kebanyakan item terjual 0–2 unit per hari, jadi kenaikan 15% di total toko
hampir tidak terlihat di level item dan tenggelam oleh noise. Data latih juga hanya punya
3–5 Easter. Fitur ini menangkap sebagian pola agregat dan tidak merusak horizon secara
keseluruhan.

**Keputusan:** fitur tetap dipakai di Fase 5, dengan status "tidak terbukti di level item".
**Pelajaran untuk Idulfitri:** dengan sedikit kejadian, manfaat fitur lebih mungkin terlihat
di level agregat, sejalan dengan rencana cadangan Fase 8.

Grafik: `reports/figures/easter_ablation.png`.

### Profil ulang hardware (36 fitur, CA_1)
| Model | `item_id` | Waktu | Ukuran model | Puncak RAM | VRAM |
|---|---|---:|---:|---:|---:|
| tweedie | kategorikal | 17 s | 86 MB | 4,6 GB | +562 MiB |
| tweedie | label | 14 s | **3,7 MB** | 4,3 GB | +452 MiB |
| 4 kuantil | label | 575 s | 100 MB | **5,0 GB** (Fase 1: 10,5 GB) | +774 MiB |

→ Label encoding menyelesaikan masalah RAM. Yang tersisa: model kuantil lambat konvergen
(belum berhenti di 3.000 iterasi, ~10 menit per toko per fold).

---

## 8. Preprocessing & kualitas data

Bagian ini menjawab pertanyaan: *"apakah data ini tidak perlu preprocessing?"*

### Preprocessing yang sudah dilakukan (tersebar di Fase 0–4)
| Langkah | Fase | Hasil |
|---|---|---|
| Cek skema, duplikat, nilai kosong | 0 | Tidak ada NaN harga, tidak ada kunci ganda |
| Cek konsistensi antar file | 0 | File validation = 1.913 hari pertama file evaluation |
| Ubah tipe data (downcast) | 1 | `int16`, `float32`, `category` |
| Ubah bentuk wide → long | 1 | 46,8 juta baris |
| Gabung harga ke penjualan | 1 | Diverifikasi dengan tes |
| Buang baris "produk belum dijual" | 1 | 12,4 juta baris dibuang |
| Tangani hari anomali (toko tutup) | 2 & 4 | Natal dikeluarkan dari training |
| Encoding kategori | 4 | Label encoding vs kategorikal |
| Feature engineering | 4 | 37 fitur |
| Indikasi stok habis | 4 | Fitur `zero_run_lag28` |

### Yang sengaja TIDAK dilakukan
| Langkah | Alasan |
|---|---|
| Scaling / normalisasi | Model pohon tidak terpengaruh skala fitur |
| Mengisi nilai kosong (imputasi) | XGBoost menangani NaN sendiri; mengisi lag kosong dengan 0 justru memberi informasi palsu |
| Membuang lonjakan penjualan | Lonjakan adalah permintaan nyata (event, promo), dan itu yang ingin diramalkan |
| Transformasi log target | `reg:tweedie` sudah dirancang untuk data hitungan yang miring dan banyak nol |

### Temuan kualitas data (`reports/eda.md` §P5)

**a) Harga anomali: kecil.**

| Pengecekan | Baris |
|---|---:|
| Harga < 0,10 dolar | 231 (26 item) |
| Harga < 20% median item-toko itu sendiri | 405 |
| Harga > 5× median item-toko itu sendiri | 34 |

Harga absolut yang rendah belum tentu salah (contoh terbanyak: item FOODS_3 yang memang
murah, 0,05 dolar dengan median 0,20). **Keputusan:** dibiarkan dan didokumentasikan,
karena hanya 0,007% data dan model pohon tahan terhadap nilai ekstrem.

**b) Deretan nol panjang: indikasi kuat stok habis (censored demand).**

| Pengecekan | Jumlah |
|---|---:|
| Item laris (laku ≥ 50% hari bukanya) | 10.004 deret |
| … yang pernah tidak terjual ≥ 28 hari beruntun | **6.235 (62%)** |
| … ≥ 90 hari beruntun | 3.117 (31%) |
| Deret yang nol terus di ≥ 28 hari terakhir sebelum d_1913 | 1.587 |
| **Porsi semua hari bernilai nol yang ada di dalam deretan ≥ 28 hari** | **29,5%** |

**Analisis:** item yang biasanya laku hampir setiap hari tidak mungkin tidak terjual 28 hari
beruntun karena kebetulan (peluang kira-kira 0,5²⁸). Deretan seperti ini hampir pasti
berarti barang **tidak tersedia di rak**. Jadi sebagian nol di data adalah *penjualan nol*,
bukan *permintaan nol*. Model yang belajar dari nol-nol ini bisa terlalu rendah memprediksi
item yang sebenarnya diminati, dan simulasi stok (Fase 7) membutuhkan permintaan, bukan
penjualan.

**Keputusan:**
1. **Sudah dikerjakan:** fitur `zero_run_lag28`, lolos tes anti-leakage. Pengecekan awal
   (CA_1): rata-rata penjualan hari ini 0,44 unit bila item sedang dalam deretan nol ≥ 28
   hari pada 28 hari lalu, dibanding 1,78 unit bila tidak, sehingga fitur ini informatif.
2. **Fase 5:** uji pembanding, yaitu baris di dalam deretan nol panjang item laris
   dikeluarkan dari training. Ambang batasnya ditetapkan sebelum melihat hasil. Evaluasi
   tetap memakai penjualan aktual.
3. **Fase 7:** simulasi stok memakai model versi tanpa deretan nol panjang.

---

## 9. Kendala yang ditemui & cara mengatasinya

| Kendala | Penyebab | Solusi |
|---|---|---|
| Unduh M5 ditolak (403) | Aturan kompetisi disetujui di akun Kaggle lain | Token API baru dari akun yang sudah menyetujui, disimpan di `~/.kaggle/access_token` |
| Token baru tidak terbaca | CLI lama (1.6.17) dan nama file salah (`acces_token.txt`) | CLI 2.2.4 di venv proyek; nama file diperbaiki |
| numpy gagal dimuat di venv | Nama folder proyek mengandung `→`; DLL Windows gagal dimuat dari path itu | Venv dipindah ke `C:\Users\ASUS\.venvs\retailfc` |
| `.venv` muncul lagi di proyek | Opsi "Create Virtual Environment" di VS Code | Dihapus; interpreter diarahkan ke venv di luar proyek |
| Objective `count:tweedie` error | Nama itu tidak ada di XGBoost | Diganti `reg:tweedie`, dokumen rencana dikoreksi |
| RAM 10,5 GB untuk model kuantil | `item_id` kategorikal + salinan model di akhir training | Diselidiki, lalu diatasi dengan label encoding (5,0 GB) |
| Natal tidak tertangkap sebagai hari tutup | Penjualannya 11–20 unit, bukan 0 | Ambang <10% penjualan normal |
| Peringatan device di prediksi | Model di GPU, data di CPU | Device booster disetel ke CPU sebelum prediksi; hasil identik |
| Commit pertama memuat nama file dokumen lama | Penggantian nama setelah push | Dua commit digabung lalu force push; riwayat bersih |

---

## 10. Keputusan yang sudah diambil

| Topik | Keputusan | Sumber |
|---|---|---|
| File penjualan | Pakai `sales_train_evaluation.csv` saja | Fase 0 |
| Baris awal | Buang hari sebelum penjualan pertama tiap item-toko | Fase 1 |
| Granularitas model | Satu model per toko, GPU | Rencana + Fase 1 |
| Hari tutup | 25 Desember: dikeluarkan dari training & rolling, forecast 0 | Fase 2 |
| Fold | 4 CV + 3 khusus + holdout, early stopping di dalam train | Fase 3 |
| Lag minimum | 28 hari (tanpa prediksi rekursif) | Fase 4 |
| Penyimpanan fitur | Tabel dasar per toko + fitur per fold saat training | Fase 4 |
| Fitur Easter | Tetap dipakai, status "tidak terbukti di level item" | Fase 4 |
| Stok habis | Fitur `zero_run_lag28` sekarang; uji keluarkan dari training di Fase 5 | §8 |
| Harga anomali | Dibiarkan, didokumentasikan | §8 |
| Encoding `item_id` | Label encoding sebagai default kerja; perbandingan final di Fase 5 | Fase 1 & 4 |
| Bahasa | Laporan, teks grafik, komentar, docstring: Indonesia; nama kode: Inggris | Permintaan pemilik |

---

## 11. Hal yang masih terbuka

| Item | Kapan |
|---|---|
| Kutip lisensi data M5 dari halaman Rules Kaggle (butuh login) | Kapan saja |
| Model kuantil lambat konvergen: tentukan `eta` atau batas iterasi | Fase 5 |
| `item_id` kategorikal vs label: bandingkan akurasi, ukuran model, dan RAM | Fase 5 |
| Setelah early stopping: latih ulang dengan semua data ≤ cutoff atau tidak | Fase 5 |
| Uji censored demand: keluarkan deretan nol panjang dari training | Fase 5 |
| Implementasi WRMSSE harus diverifikasi terhadap referensi | Fase 5 |
| Data apotek Indonesia belum diunduh dan belum diaudit | Fase 8 |

---

## 12. Riwayat commit & isi repo

| Commit | Isi |
|---|---|
| `934eede` | Fase 0: setup proyek dan verifikasi data M5 |
| `ba163a0` | Fase 1: format panjang hemat memori dan profil hardware |
| `ce95f2f` | Fase 2: EDA terarah |
| `bd569fb` | Fase 3: desain validasi + terjemahan ke bahasa Indonesia |
| *(belum)* | Fase 4: fitur, ablation Easter, kualitas data, profil ulang, recap ini |

Semua commit atas nama RizkyLukman. Data mentah, Parquet, dan token Kaggle tidak masuk repo.

### Modul kode (`src/retailfc/`)
| File | Fungsi | Fase |
|---|---|---|
| `verify_m5.py` | Verifikasi skema & isi data | 0 |
| `io.py` | Format panjang per toko → Parquet | 1 |
| `profiling.py` | Alat ukur RAM/VRAM | 1 |
| `hardware_profile.py` | Profil training 1 toko | 1 & 4 |
| `eda.py` | EDA terarah + kualitas data + grafik | 2 |
| `folds.py` | Definisi & validasi fold | 3 |
| `leakage.py` | Pengecekan anti-leakage | 3 |
| `calendar_features.py` | Fitur kalender & hari libur bergerak | 4 |
| `features.py` | Fitur lag, rolling, stok habis, harga, target encoding | 4 |
| `train.py` | Training per toko per fold (akan diperluas) | 4–5 |
| `ablation_easter.py` | Uji fitur Easter | 4 |

### Tes (`tests/`)
| File | Isi | Jumlah |
|---|---|---:|
| `test_io.py` | Output Fase 1 vs data asli | 4 |
| `test_folds.py` | Aturan desain fold | 11 |
| `test_leakage.py` | Alat anti-leakage menangkap fitur bocor | 8 |
| `test_features.py` | Anti-leakage & kebenaran setiap fitur | 38 |
| **Total** | | **61** |

---

## 13. Langkah berikutnya

**Fase 5 — Model & baseline** (training utama):
1. Baseline wajib: seasonal naive, moving average 28 hari, Croston/TSB, ETS.
2. XGBoost `reg:tweedie` (point) dan model kuantil, 10 toko × 4 fold CV.
3. Metrik: WRMSSE (diverifikasi dulu), RMSSE, MAE, bias, pinball loss.
4. Memutuskan hal-hal terbuka di §11 (encoding, kuantil, latih ulang, censored demand).

Setelah itu: Fase 6 (error per segmen + SHAP), Fase 7 (simulasi stok), Fase 8 (data
Indonesia), Fase 9 (README akhir).
