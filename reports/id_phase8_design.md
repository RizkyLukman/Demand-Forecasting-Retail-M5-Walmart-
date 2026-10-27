# Fase 8 Langkah 2: Desain Analisis (ditetapkan sebelum hasil dihitung)

> **Status (28 Okt 2026):** desain saja, belum ada hasil. Parameter ada di
> `config.yaml → id_pharmacy`. Kalender: `data/external/id_calendar.csv`
> (`src/retailfc/id_calendar.py`). Desain ini mengikuti jalur "tidak cukup" dari
> `reports/id_data_audit.md`.
>
> **Catatan kejujuran:** desain ini disusun setelah audit, jadi beberapa pola sudah terlihat
> (Senin sepi, penurunan setelah Lebaran pada 28 Jul–1 Agu). Untuk mencegah pemilihan jendela
> yang menguntungkan, jendela di bawah berpatokan pada **tanggal resmi** (1 Ramadan, cuti
> bersama, Idulfitri) dengan panjang tetap 14 hari, bukan pada pola yang terlihat.

---

## 1. Data yang dipakai

- Baris `det_sales` yang punya header di `ms_sales`. Tanggal diambil dari header nota.
- Periode: **1 Apr–31 Des 2015**.
- Baris yang sama persis tetap dipakai (obat yang sama bisa muncul dua kali di satu nota).
  Retur (QTY negatif) dihitung sebagai pengurang. Unit harian per produk dipotong di 0.
- Jenis nota: `RJ`, `RI`, dan total. Jenis lain (RL, KC, B, PR) hanya masuk ke total.

## 2. Studi efek Ramadan/Lebaran

**Metrik utama:** jumlah nota per hari, untuk total, RJ, dan RI. Metrik kedua: unit per hari
(total). Nota dipilih sebagai metrik utama karena tidak terpengaruh QTY pecahan atau retur.

**Jendela patokan** (inklusif, dari tanggal resmi):

| Jendela | Tanggal | Patokan |
|---|---|---|
| `ramadan_w1_2` | 18 Jun–1 Jul | 1 Ramadan + 13 hari |
| `ramadan_w3_4` | 2–15 Jul | 14 hari berikutnya, sampai sebelum cuti bersama |
| `lebaran_cuti` | 16–21 Jul | cuti bersama 16 Jul s.d. cuti bersama 21 Jul |
| `post_1` | 22 Jul–4 Agu | 14 hari setelah cuti bersama terakhir |
| `post_2` | 5–18 Agu | 14 hari berikutnya |

Periode event = 18 Jun–18 Agu (62 hari). Jendela 14 hari dipecah menjadi **jendela mingguan**
(9 jendela: Ramadan minggu 1–4, `lebaran_cuti` 6 hari, pasca Lebaran minggu 1–4).

**Nilai harapan (baseline): level lokal dengan margin 28 hari.**
1. Faktor hari dalam minggu dihitung dari semua hari biasa di luar periode event. Hari biasa
   artinya bukan hari libur atau cuti bersama di `id_calendar.csv`.
2. Level dihitung dari rata-rata deret yang pola mingguannya sudah dihapus, pada 28 hari biasa
   terdekat **sebelum** dan 28 hari biasa terdekat **sesudah** periode event. Level ini
   diinterpolasi linear di sepanjang periode event.
3. Nilai harapan = level × faktor hari dalam minggu.

**Efek** = total aktual di jendela ÷ total nilai harapan − 1.

**Uji plasebo** (pengganti uji statistik, karena hanya ada satu Ramadan): periode event palsu
sepanjang 62 hari digeser di data Agu–Des (60 posisi). Model baseline dilatih tanpa periode
palsu itu lalu dipakai menebak isinya, dan efek dihitung di jendela dengan posisi relatif
yang sama. Hari libur di periode palsu dikeluarkan dari jumlah. Dilaporkan: efek di jendela
asli, persentil 5–95 efek plasebo, dan peringkat efek asli di antara plasebo. Efek disebut
**"lebih besar dari fluktuasi biasa"** hanya jika berada di luar persentil 5–95 plasebo.

### 2.1 Kalibrasi baseline dan panjang jendela (sudah dijalankan)

Pilihan di atas ditentukan **hanya dari data di luar periode event**. Efek Ramadan/Lebaran
belum dihitung saat memilih. Detail: `reports/id_event_calibration.json`.

| Baseline | RMS log-rasio plasebo (rata-rata total/RJ/RI) |
|---|---:|
| median per hari dalam minggu (semua hari biasa) | 0,064 |
| log-linear: hari dalam minggu + tren linear | 0,079 |
| level lokal, margin 14 hari | 0,062 |
| **level lokal, margin 28 hari (dipilih)** | **0,061** |
| level lokal, margin 42 hari | 0,062 |

Selisih antar metode level lokal dan median kecil (0,061–0,064). Hanya model log-linear
yang jelas lebih buruk.

| Panjang jendela | RMS total | RMS RJ | RMS RI |
|---|---:|---:|---:|
| 14 hari | 0,039 | 0,090 | 0,053 |
| 7 hari (dipilih) | 0,045 | 0,092 | 0,062 |

Jendela mingguan hanya sedikit lebih berisik, tetapi resolusinya dua kali lebih baik.

**Keterbatasan kalibrasi:**
- 60 posisi plasebo saling tumpang tindih. Data Agu–Des hanya memuat ±2 periode 62 hari
  yang benar-benar terpisah, jadi pita persentil 5–95 bersifat kasar.
- Plasebo RJ condong negatif (rata-rata log-rasio −0,07, persentil 5–95 efek −14% s.d.
  +1%). Artinya baseline cenderung menebak RJ terlalu tinggi di Agu–Des. Karena efek asli
  dinilai relatif terhadap pita plasebo, bias ini ikut diperhitungkan.
- Dengan 9 jendela dan pita 5–95, kira-kira 1 jendela bisa "keluar pita" hanya karena
  kebetulan. Kesimpulan tidak boleh bertumpu pada satu jendela saja.

## 3. Forecast agregat (horizon 14 hari)

**Target:**
1. Nota per hari untuk RJ dan RI, serta unit per hari (total), dengan satu model gabungan
   untuk ketiga deret.
2. Unit per hari per produk, untuk produk yang terjual di ≥ 50% hari di data train fold itu
   (±200 produk).

**Fold:**
- **CV utama:** 8 cutoff setiap 14 hari, 10 Sep s.d. 17 Des (horizon 11 Sep–31 Des).
- **Fold khusus**, dilaporkan terpisah dan tidak dirata-rata ke skor utama:
  `lebaran` (cutoff 9 Jul, horizon 10–23 Jul) dan `post_lebaran` (cutoff 23 Jul,
  horizon 24 Jul–6 Agu).
- Train selalu mulai 1 Apr. Early stopping memakai 14 hari terakhir **di dalam** train.
- Fitur hanya memakai data ≤ cutoff (tes anti-leakage seperti di M5).

**Fitur:** hari dalam minggu, tanggal dalam bulan, `day_off` (libur/cuti bersama dari
`id_calendar.csv`), hari menuju dan sejak libur terdekat (dipotong ±7), lag 14/21/28,
rolling mean 7/14/28 dari lag 14, kode deret/produk (label encoding), dan rata-rata target
per deret dari data train fold.

- **Fitur Ramadan** (`is_ramadan`, `ramadan_day`, `days_to_idulfitri`) **tidak dipakai di
  CV utama**, karena horizon Sep–Des tidak memuat Ramadan, dan di fold `lebaran` data train
  belum pernah melihat Ramadan. Ini keterbatasan yang dicatat, bukan pilihan.
- Yang diuji di fold `lebaran`: apakah fitur libur **umum** (`day_off`), yang dipelajari dari
  libur lain (1 Mei, Kenaikan, Waisak, dll.), bisa memprediksi Lebaran.

**Model:** XGBoost `reg:tweedie` (1,1), `max_depth` 4, `eta` 0,05, di CPU. Parameter
ditetapkan di awal dan **tidak di-tuning**, karena data terlalu kecil untuk tuning sekaligus
evaluasi yang jujur. Tidak ada holdout terpisah. Karena itu CV di sini berfungsi sebagai
evaluasi akhir.

**Baseline:**
- pengulangan minggu terakhir (seasonal naive 7 hari);
- rata-rata per hari dalam minggu dari 4 minggu terakhir;
- MA 28 hari;
- ETS musiman 7 (`statsforecast`), untuk deret agregat;
- Croston/TSB, untuk produk.

**Metrik:** RMSSE per deret (skala = error naive 7 hari di train), WAPE, dan bias. Untuk
produk: rata-rata RMSSE berbobot omzet 28 hari terakhir train (mirip WRMSSE). Tidak ada MAPE.

## 4. Perbandingan dengan Easter di M5 (Langkah 3)

Tabel kualitatif: arah efek, durasi, kelompok yang terdampak, dan apakah fitur hari libur
membantu forecast. Bahan dari `reports/easter_ablation.md` dan `reports/eda.md`.

## 5. Yang sengaja tidak dilakukan

- Tidak ada forecast per produk dengan fitur Ramadan.
- Tidak ada ablation dengan vs tanpa fitur Ramadan (butuh minimal 2 Ramadan).
- Tidak ada lag 364 atau musiman tahunan.
- Tidak ada tuning parameter.
