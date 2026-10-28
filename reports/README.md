# Daftar Laporan

Laporan disusun mengikuti urutan pengerjaan proyek, dari pengecekan data sampai transfer ke data
Indonesia. Setiap laporan punya tautan ke laporan sebelum dan sesudahnya di bagian atas.
Ringkasan hasil keseluruhan ada di [README utama](../README.md).

| No | Laporan | Pertanyaan | Temuan utama |
|---|---|---|---|
| 0 | [Data dan sumber](../data/SOURCES.md) | Apakah data M5 sesuai dokumentasi? | Skema dan rentang hari cocok; bobot WRMSSE cocok dengan bobot resmi untuk 42.840 deret |
| 1 | [Profil hardware](01_hardware_profile.md) | Apakah pipeline muat di laptop 16 GB / GPU 4 GB? | RAM host yang membatasi, bukan VRAM. Label encoding dan melepas tabel sebelum training menurunkan puncak RAM ke ±4,4 GB per toko |
| 2 | [EDA](02_eda.md) | Seperti apa pola nol, mingguan, SNAP, hari tutup, dan Easter? | 59,6% hari bernilai nol; 25 Des praktis tutup; efek Easter ±3 hari; 62% item laris pernah tidak terjual ≥ 28 hari |
| 3 | [Desain validasi](03_validation_design.md) | Bagaimana menguji model dengan jujur? | 4 fold rolling origin + 3 fold khusus; holdout d_1914–d_1941 disimpan untuk akhir |
| 4 | [Ablation Easter](04_easter_ablation.md) | Apakah fitur jarak ke Easter membantu? | Tidak terbukti di level item; membantu di total harian FOODS |
| 5a | [Model dan baseline](05a_models.md) | Seberapa akurat XGBoost dibanding baseline? | WRMSSE 0,582 vs 0,916 (classical), menang di 7 dari 7 fold; di level item keunggulannya hanya 2–3% |
| 5b | [Model kuantil final](05b_quantile_cap1000.md) | Model kuantil mana untuk simulasi stok? | eta 0,2, batas 1.000 iterasi; coverage P80/P90 = 81,0%/89,4% |
| 5c | [Uji batas iterasi](05c_quantile_cap3000.md) | Apakah iterasi lebih banyak membantu? | Tidak: validasi membaik, ramalan sedikit memburuk |
| 5d | [Tuning parameter pohon](05d_tree_tuning.md) | Apakah parameter pohon bisa diperbaiki? | Hanya min_child_weight 200 yang lolos di 2 toko; di 10 toko −0,003 (< noise), jadi tidak diganti |
| 6 | [Analisis error](06_error_analysis.md) | Di mana XGBoost gagal? | Kalah di total segmen erratic dan akhir pekan per item; terlalu bergantung pada rata-rata item lama (SHAP 42%) |
| 6b | [Rata-rata item jendela terbaru](06b_te_recent_window.md) | Apakah rata-rata item 182/365 hari memperbaiki item yang sedang berubah? | Bias kelompok berkurang, tetapi WRMSSE memburuk; fitur tidak diganti |
| 7 | [Simulasi stok](07_inventory_sim.md) | Apakah forecast lebih akurat membuat stok lebih murah? | XGBoost efisien di layanan menengah; di fill rate ≥ 98% metode empiris termurah (10 dari 12 asumsi biaya) |
| 8a | [Audit data apotek](08a_id_data_audit.md) | Apakah data apotek cukup untuk fitur Ramadan? | Tidak: hanya 2015 (lengkap Apr–Des), satu Idulfitri |
| 8b | [Desain analisis apotek](08b_id_design.md) | Apa yang masih bisa diuji dengan jujur? | Studi efek dengan uji plasebo + forecast 14 hari; ditetapkan sebelum hasil dihitung |
| 8c | [Transfer ke apotek](08c_id_transfer.md) | Bagaimana efek Lebaran dan kinerja XGBoost di data kecil? | Pekan Lebaran normal, nota turun 25%/37% dua minggu sesudahnya; XGBoost kalah dari baseline sederhana |
| 9 | [Evaluasi holdout M5](09_holdout.md) | Apakah hasil CV bertahan di 28 hari yang tidak pernah disentuh? | Ya: XGBoost 0,563 (CV 0,52–0,64), 35% lebih kecil dari baseline terbaik; empiris tetap termurah untuk stok |

**Folder lain:**
- [`figures/`](figures/): semua grafik. Nama file diawali fase (`eda_`, `phase5_`, `phase8_`, dst.).
- [`data/`](data/): angka hasil (JSON/CSV) yang dikutip laporan.
- [`runs/`](runs/): config, metrik, dan info training setiap run model final.
- [`archive/`](archive/): rekap hari pertama (23 Okt 2026), tidak diperbarui.
