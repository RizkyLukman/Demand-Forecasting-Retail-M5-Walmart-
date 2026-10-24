# Demand Forecasting Retail: M5 Walmart → Indonesia

Proyek ini berawal dari satu pertanyaan sederhana: kalau penjualan harian setiap produk di setiap
toko diramal dengan XGBoost, seberapa akurat hasilnya, di bagian mana modelnya gagal, dan apakah
ramalan yang lebih akurat benar-benar membuat keputusan stok lebih murah?

Saya memakai data kompetisi M5 dari Walmart (30.490 kombinasi produk-toko, 1.941 hari). Rencananya,
setelah itu pipeline yang sama akan saya coba di data penjualan apotek di Indonesia, dengan fitur
Ramadan dan Idulfitri.

Saya sengaja tidak berhenti di skor WRMSSE. Notebook M5 di Kaggle sudah ribuan, jadi yang ingin
saya tunjukkan di sini adalah prosesnya: backtest yang jujur, analisis error per segmen, dan
simulasi stok yang menghubungkan forecast dengan biaya.

**Status:** Fase 0–7 selesai. Fase 8 (data Indonesia) belum dimulai. Holdout terakhir
(25 Apr–22 Mei 2016) sengaja **belum dipakai sama sekali**. Holdout baru akan dibuka sekali di akhir
proyek, supaya ada satu angka akhir yang benar-benar tidak ikut memengaruhi keputusan saya selama
eksperimen.

---

## Ringkasan hasil

- **XGBoost jauh lebih akurat daripada semua baseline.** WRMSSE rata-rata 4 fold **0,582**,
  sedangkan baseline terbaik (ETS + TSB) **0,916**. Error-nya sekitar 36% lebih kecil, dan
  XGBoost menang di ketujuh periode uji.
- **Tetapi keunggulannya ada di level gabungan.** Untuk total penjualan harian 10 toko, akurasinya
  96,4% (baseline 90,4%). Untuk satu produk di satu toko per hari, akurasinya hanya 22,3%, sedikit
  di bawah baseline (24,2%). Penjualan per produk per hari memang sangat acak.
- **Model terlalu bergantung pada rata-rata lama produk.** Dari SHAP, fitur terpentingnya adalah
  rata-rata penjualan item sepanjang riwayat (42%). Akibatnya, produk yang sedang naik daun
  diramal terlalu rendah (−7% sampai −32%), dan produk yang sedang turun terlalu tinggi.
- **Forecast yang lebih akurat tidak otomatis membuat stok lebih murah.** Dalam simulasi stok,
  XGBoost paling efisien untuk target layanan menengah. Tetapi kalau kehabisan stok mahal dan
  targetnya fill rate di atas 98%, metode empiris sederhana dari riwayat 1 tahun justru paling
  murah.

---

## Data

| | |
|---|---|
| Sumber | [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), Kaggle |
| Isi | 3.049 produk, 10 toko (CA, TX, WI), 3 kategori, 7 departemen |
| Periode | 29 Jan 2011 – 22 Mei 2016 (d_1 – d_1941) |
| Ukuran | 59,2 juta baris setelah diubah ke format panjang; 46,8 juta setelah membuang hari sebelum produk pertama kali dijual |

Data tidak disertakan di repo ini. Untuk mengunduhnya, perlu akun Kaggle dan persetujuan aturan
kompetisi. Checksum file dan hasil pengecekan skemanya ada di [`data/SOURCES.md`](data/SOURCES.md).

Beberapa hal yang saya temukan saat EDA dan ikut menentukan desain model:

- 59,6% hari produk-toko bernilai nol.
- 25 Desember praktis tutup, walaupun penjualannya tidak persis nol.
- Efek SNAP (bantuan pangan) jelas terlihat di kategori FOODS.
- 62% produk laris pernah tidak terjual ≥ 28 hari berturut-turut. Kemungkinan besar ini stok habis,
  bukan karena tidak ada pembeli.

---

## Cara saya menguji

Satu holdout saja menurut saya tidak cukup, karena hasilnya bisa kebetulan bagus di satu periode.
Jadi saya memakai **4 fold rolling origin**. Setiap fold meramal 28 hari ke depan:

| Fold | Periode yang diramal |
|---|---|
| cv1 | 4–31 Jan 2016 |
| cv2 | 1–28 Feb 2016 |
| cv3 | 29 Feb–27 Mar 2016 |
| cv4 | 28 Mar–24 Apr 2016 |

Ditambah tiga fold khusus (Easter 2014, Easter 2015, dan akhir tahun 2015) yang dilaporkan terpisah.

Aturan yang saya pegang selama proyek:

- Semua fitur hanya memakai data sampai cutoff. Ini dicek dengan tes anti-leakage di beberapa
  cutoff, bukan diasumsikan.
- Early stopping memakai 28 hari terakhir sebelum cutoff, bukan periode yang diramal.
- Setiap eksperimen punya aturan keputusan yang saya tulis **sebelum** melihat hasilnya.
- WRMSSE saya cocokkan dengan bobot resmi M5 untuk ke-42.840 deret (selisih maksimum 2,3e-9)
  sebelum angkanya saya percaya.

---

## Hasil 1: akurasi forecast

![WRMSSE per fold](reports/figures/phase5_wrmsse_by_fold.png)

| Model | WRMSSE (rata-rata 4 fold ± sd) |
|---|---:|
| **XGBoost** (satu model per toko, objective tweedie) | **0,582 ± 0,051** |
| Classical (ETS untuk deret stabil, TSB untuk deret intermiten) | 0,916 ± 0,032 |
| Seasonal naive (nilai 7 hari lalu) | 0,986 ± 0,101 |
| TSB | 1,100 |
| Moving average 28 hari | 1,103 |
| Croston | 1,128 |

WRMSSE lebih kecil lebih baik. Nilai 1 kira-kira setara dengan menebak "besok sama dengan hari ini".

![Akurasi per level](reports/figures/phase5_accuracy_by_level.png)

Grafik ini yang paling menjelaskan karakter model. Di level total, toko, kategori, dan
departemen, XGBoost unggul 5–6 poin. Di level produk, semua model hampir sama, dan sama-sama rendah.

Salah satu keputusan yang paling membantu adalah **mengeluarkan deretan nol panjang dari data
training** (yang kemungkinan besar stok habis), supaya model belajar permintaan, bukan kekosongan
rak. WRMSSE turun dari 0,756 ke 0,690 di eksperimen 2 toko.

Untuk keperluan stok, saya juga melatih model kuantil (P10/P50/P80/P90). P80 dan P90-nya tepat
sasaran: aktual berada di bawah P80 pada 81,0% hari dan di bawah P90 pada 89,4% hari.

---

## Hasil 2: di mana model lemah

![Per segmen](reports/figures/phase6_segments.png)

Saya mengelompokkan setiap deret dengan skema Syntetos–Boylan (smooth, erratic, intermittent,
lumpy).

- Per produk, XGBoost hanya 1–3% lebih baik daripada baseline terbaik di semua segmen.
- Saat dijumlahkan, error-nya saling menetralkan, sehingga total segmen intermittent dan lumpy
  sekitar 54% lebih akurat.
- **XGBoost tidak lebih baik** untuk total segmen erratic (cenderung terlalu tinggi, bias +9%),
  untuk hari Sabtu–Minggu di level produk, dan di fold Easter 2015 di level produk.

![SHAP global](reports/figures/phase6_shap_global.png)

Dari SHAP, model pada dasarnya bekerja sebagai "rata-rata item jangka panjang + penjualan terbaru +
pola hari dalam minggu". Harga hanya menyumbang 3%, dan fitur event (termasuk Easter) hampir tidak
dipakai (0,3%).

![Item yang tumbuh](reports/figures/phase6_shap_cases.png)

Contoh paling jelas: FOODS_1_096 di toko WI_2 sedang laku ±42 per hari, tetapi rata-rata
riwayatnya hanya 13 per hari. XGBoost meramal 35 per hari, sedangkan aktualnya 57. Pola ini
sistematis di seluruh produk yang sedang tumbuh atau turun.

Saya sempat menimbang untuk memperbaiki fitur ini, tetapi hitungan saya menunjukkan perbaikan
WRMSSE totalnya kemungkinan di bawah 1%. Jadi eksperimennya saya tunda dan temuannya saya catat.

---

## Hasil 3: dampak ke keputusan stok

Ini bagian yang paling ingin saya jawab. Saya mensimulasikan kebijakan stok sederhana:

- stok dicek seminggu sekali, dan pesanan tiba 2 hari kemudian;
- target stok = kuantil permintaan selama 9 hari;
- penjualan yang tidak terlayani dianggap hilang.

M5 tidak punya data biaya, jadi biayanya saya **asumsikan** (simpan 25% per tahun, kehabisan
30% dari harga per unit), lalu saya uji dengan 12 kombinasi lain.

![Kurva trade-off stok](reports/figures/phase7_frontier.png)

Grafik ini membandingkan fill rate pada jumlah stok yang sama:

- Sampai sekitar 10 hari stok, **forecast XGBoost + safety stock paling efisien**, yaitu 1–3 poin
  fill rate lebih tinggi untuk stok yang sama.
- Di atas fill rate ±97%, **metode empiris** (kuantil penjualan 9 hari dari riwayat 1 tahun)
  paling efisien. Asumsi distribusi normal tidak sanggup menangkap lonjakan penjualan produk yang
  jarang laku.

![Sensitivitas biaya](reports/figures/phase7_cost_sensitivity.png)

Karena kehabisan stok jauh lebih mahal daripada menyimpan, kebijakan termurah hampir selalu
mengejar fill rate tinggi. Di wilayah itu, metode empiris termurah di 10 dari 12 asumsi biaya.
XGBoost hanya menang ketika biaya kehabisan stok rendah.

Dua hal lain yang saya pelajari:

- Menjumlahkan kuantil harian bukan cara yang tepat untuk mendapat kuantil total 9 hari. P50-nya
  hanya mencakup 28% periode, sedangkan P90-nya 97%.
- Metode gabungan (level dari XGBoost, bentuk distribusi dari riwayat) yang saya harapkan bisa
  mengambil kelebihan keduanya ternyata lebih mahal di semua asumsi biaya.

---

## Yang tidak berhasil

Saya catat juga eksperimen yang gagal, karena ini bagian dari hasilnya:

| Percobaan | Hasil |
|---|---|
| `item_id` sebagai fitur kategorikal | Lebih buruk (0,808 vs 0,756), dan model 18× lebih besar |
| Melatih ulang setelah early stopping | Tidak membantu secara konsisten |
| Tweedie power 1,2–1,5, Poisson, squared error | Semua kalah dari tweedie 1,1 |
| Fitur hari libur bergerak (Easter) | Tidak terbukti membantu di level produk, hanya di total FOODS |
| Menaikkan batas iterasi model kuantil dari 1.000 ke 3.000 | Skor validasi membaik, tetapi ramalan sedikit memburuk (+0,07%) |
| Metode stok gabungan XGBoost + empiris | Lebih mahal di 12 dari 12 asumsi biaya |

Satu catatan teknis yang mungkin berguna bagi orang lain: di xgboost 3.4.1, SHAP versi GPU
(`pred_contribs`) memberi hasil yang tidak aditif untuk model dengan fitur kategorikal (selisih
sampai 0,57 pada 98% baris). Versi CPU tepat. Semua angka SHAP di sini dihitung di CPU.

---

## Keterbatasan

- Semua angka berasal dari backtest. Holdout belum dipakai.
- Penjualan tidak sama dengan permintaan. Saat stok habis, penjualan tercatat lebih rendah dari
  permintaan sebenarnya, dan M5 tidak punya data persediaan untuk memastikannya.
- Biaya di simulasi stok adalah asumsi. Kedaluwarsa, kapasitas rak, dan minimum order tidak
  dimodelkan.
- Parameter pohon XGBoost (depth, min_child_weight, dan lainnya) belum saya tuning. Yang saya uji
  baru objective, encoding, dan perlakuan data.
- Fitur harga di SHAP menunjukkan bagaimana model memakai harga, bukan efek harga terhadap
  permintaan.

---

## Laporan lengkap

| Laporan | Isi |
|---|---|
| [`reports/eda.md`](reports/eda.md) | EDA: hari nol, pola mingguan, SNAP, hari tutup, Easter, kualitas data |
| [`reports/validation_design.md`](reports/validation_design.md) | Desain fold dan aturan anti-leakage |
| [`reports/easter_ablation.md`](reports/easter_ablation.md) | Uji fitur hari libur bergerak |
| [`reports/phase5_models.md`](reports/phase5_models.md) | Baseline, XGBoost, model kuantil, dan cara membaca WRMSSE |
| [`reports/quantile_cap1000.md`](reports/quantile_cap1000.md), [`reports/quantile_cap3000_comparison.md`](reports/quantile_cap3000_comparison.md) | Uji batas iterasi model kuantil |
| [`reports/error_analysis.md`](reports/error_analysis.md) | Error per segmen, SHAP, dan studi kasus |
| [`reports/inventory_sim.md`](reports/inventory_sim.md) | Simulasi stok dan analisis sensitivitas biaya |
| [`reports/hardware_profile.md`](reports/hardware_profile.md) | Pemakaian RAM/VRAM dan cara menghematnya |

---

## Menjalankan ulang

Semua dikerjakan di laptop biasa (RAM 16 GB, GPU RTX 3050 4 GB), jadi pipeline-nya sengaja
diproses per toko. Python 3.12, xgboost 3.4.1.

```powershell
# venv saya taruh di luar folder proyek, karena karakter "→" di path membuat numpy gagal
# memuat DLL di Windows
python -m venv $HOME\.venvs\retailfc
& $HOME\.venvs\retailfc\Scripts\Activate.ps1
pip install -e ".[dev]"

kaggle competitions download -c m5-forecasting-accuracy   # ekstrak ke data/raw/m5/

python -m retailfc.verify_m5        --config config.yaml
python -m retailfc.io               --config config.yaml
python -m retailfc.features         --config config.yaml
python -m retailfc.baselines        --config config.yaml
python -m retailfc.xgb_final        --config config.yaml
python -m retailfc.quantile_final   --config config.yaml
python -m retailfc.error_analysis   --config config.yaml
python -m retailfc.shap_analysis    --config config.yaml
python -m retailfc.inventory_sim    --config config.yaml
pytest -q
```

Semua parameter dan asumsi ada di [`config.yaml`](config.yaml). Rencana kerja dan aturan
proyeknya ada di [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

---

## Selanjutnya

1. **Fase 8:** audit data penjualan apotek di Indonesia (Mendeley Data, DOI
   `10.17632/2ym7v78wtd.1`, CC BY 4.0). Kalau datanya cukup, pipeline yang sama dipakai dengan
   fitur Ramadan dan Idulfitri. Kalau tidak cukup, cakupannya saya turunkan menjadi studi efek
   Ramadan di level agregat.
2. **Holdout:** evaluasi forecast dan simulasi stok di 25 Apr–22 Mei 2016, sekali saja.
3. **Fase 9:** rangkuman akhir.
