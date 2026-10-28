# Fase 6b: Eksperimen Rata-rata Item Jendela Terbaru

*[← 06 Analisis error](06_error_analysis.md) · [Daftar laporan](README.md) · [07 Simulasi stok →](07_inventory_sim.md)*

> **Status (29 Okt 2026):** selesai. **Fitur tidak diganti.** Kode: `src/retailfc/te_recent_experiment.py`
> (fitur di `features.fold_features(..., recent_windows)`). Angka lengkap:
> `reports/data/te_recent_experiment.json`. Holdout tidak disentuh.

---

## Ringkasan

- **Pertanyaan** (ditunda dari Fase 6 §8.6): `te_item_mean` adalah rata-rata penjualan item
  sepanjang riwayat train, dan menjadi fitur terpenting model (SHAP 42%). Apakah versi jendela
  terbaru membuat model lebih cepat menyesuaikan diri dengan item yang sedang naik atau turun?
- **Jawaban: tidak untuk akurasi total.** Ketiga varian lebih buruk dari model sekarang di WRMSSE,
  jadi tidak ada yang lanjut ke konfirmasi 10 toko.
- **Tetapi bias per kelompok berkurang** kalau rata-rata seluruh riwayat diganti dengan rata-rata
  365 hari. Bias item yang turun tajam turun dari +12% ke +5%, dan bias item yang tumbuh sangat
  cepat dari −31% ke −26%. Harganya, WRMSSE memburuk 0,690 → 0,696.
- Artinya, kebiasaan model menarik ramalan ke rata-rata lama sebagian memang benar: item yang
  sedang naik atau turun sering kembali ke levelnya. Mengikuti level terbaru memperbaiki kelompok
  yang berubah, tetapi merugikan kelompok lain lebih banyak.

## 1. Desain (ditetapkan sebelum run)

| Varian | Isi |
|---|---|
| `base` | konfigurasi final saat ini (`model.xgb_point`) |
| `add_365` | + `te_item_mean_365` (rata-rata 365 hari terakhir sampai cutoff, hari tutup dikeluarkan) |
| `add_182` | + `te_item_mean_182` |
| `replace_365` | `te_item_mean` diganti `te_item_mean_365` |

- Data: CA_3 + WI_2 × cv1–cv4, sama dengan Fase 5 dan tuning.
- tau = noise antar-seed dari tuning (0,00705).
- Aturan menerima: varian terbaik harus lebih baik dari base > tau **dan** lebih baik di ≥ 3 dari
  4 fold, lalu lolos aturan yang sama di 10 toko × 4 fold.
- Bias per kelompok pertumbuhan (definisi Fase 6 §8.5) hanya dilaporkan, tidak dipakai untuk
  memutuskan.
- Fitur baru lolos tes anti-leakage di 5 cutoff (`tests/test_features.py`). Run `base`
  menghasilkan 0,69025, sama persis dengan tuning, jadi kode baru tidak mengubah perilaku lama.

## 2. Hasil

| Varian | WRMSSE | cv1 | cv2 | cv3 | cv4 | Fold lebih baik dari base |
|---|---:|---:|---:|---:|---:|---:|
| **base** | **0,69025** | 0,6421 | 0,8174 | 0,6825 | 0,6190 | – |
| add_365 | 0,69179 | 0,6486 | 0,8058 | 0,7081 | 0,6047 | 2/4 |
| replace_365 | 0,69585 | 0,6461 | 0,8183 | 0,6920 | 0,6270 | 0/4 |
| add_182 | 0,70596 | 0,6447 | 0,7917 | 0,6979 | 0,6896 | 1/4 |

Varian terbaik (`add_365`) lebih buruk 0,0015 dari base → **ditolak**. Konfirmasi 10 toko tidak
dijalankan.

### Bias per kelompok pertumbuhan (2 toko, total 4 fold)

Bias = total ramalan ÷ total aktual − 1. Kelompok = rata-rata 56 hari terakhir sebelum cutoff ÷
rata-rata sejak penjualan pertama.

| Kelompok | base | add_365 | add_182 | replace_365 |
|---|---:|---:|---:|---:|
| < 0,5 (turun tajam) | +12,1% | +11,1% | +10,0% | **+5,2%** |
| 0,5–0,8 (turun) | +5,5% | +5,2% | +5,6% | **+2,2%** |
| 0,8–1,25 (stabil) | −3,6% | −3,5% | −3,5% | −3,8% |
| 1,25–2 (tumbuh) | −6,3% | −6,4% | −5,9% | **−4,4%** |
| 2–4 (tumbuh cepat) | −13,0% | −13,6% | −12,3% | **−9,1%** |
| > 4 (tumbuh sangat cepat) | −31,0% | −34,2% | −31,5% | **−26,3%** |

- **Menambah** fitur jendela terbaru hampir tidak mengubah bias. Model tetap memilih rata-rata
  jangka panjang.
- **Mengganti** fitur itu mengurangi bias di kelima kelompok yang berubah. Penurunannya 15% dari
  bias semula di kelompok tumbuh sangat cepat, sampai 60% di kelompok turun (0,5–0,8). Tetapi
  bias masih jauh dari nol, dan WRMSSE memburuk.
- Angka di sini dari 2 toko, jadi tidak sama persis dengan tabel 10 toko di Fase 6 §8.5. Polanya
  sama.

## 3. Kesimpulan dan dampak

- Konfigurasi final **tidak berubah**, jadi model kuantil dan simulasi stok Fase 7 tidak perlu
  dijalankan ulang.
- Bias terhadap item yang sedang tumbuh tetap menjadi **keterbatasan yang dicatat**. Untuk
  keputusan stok, kelompok tumbuh cepat sudah ditandai sebagai kelompok berisiko di Fase 7, dan
  di kelompok itu classical + safety stock paling murah.
- Kemungkinan perbaikan yang **tidak** diuji di sini: model terpisah untuk item yang sedang
  berubah, atau koreksi pasca-prediksi berdasarkan rasio pertumbuhan. Keduanya butuh aturan
  pemilihan baru dan berisiko overfit ke kelompok kecil.
