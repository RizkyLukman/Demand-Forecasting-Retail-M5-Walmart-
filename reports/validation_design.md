# Fase 3 — Desain validasi (P1)

Kode: `src/retailfc/folds.py`, `src/retailfc/leakage.py`. Tabel fold: `reports/folds.csv`
(`python -m retailfc.folds --config config.yaml`). Tes: `tests/test_folds.py`,
`tests/test_leakage.py`.

## Fold

Horizon 28 hari. Training dan fitur hanya boleh memakai hari ≤ cutoff. Early stopping
memakai **28 hari terakhir sebelum cutoff**, di dalam jendela train, tidak pernah horizon.

| Fold | Peran | Cutoff | Early stopping | Horizon | Tanggal horizon | Event di horizon |
|---|---|---|---|---|---|---|
| cv1 | cv | d_1801 | d_1774–d_1801 | d_1802–d_1829 | 2016-01-04 → 01-31 | MLK Day, Orthodox Christmas |
| cv2 | cv | d_1829 | d_1802–d_1829 | d_1830–d_1857 | 2016-02-01 → 02-28 | Super Bowl, Valentine's, Presidents Day, Lent |
| cv3 | cv | d_1857 | d_1830–d_1857 | d_1858–d_1885 | 2016-02-29 → 03-27 | St Patrick's, Purim End, **Easter (hari terakhir)** |
| cv4 | cv | d_1885 | d_1858–d_1885 | d_1886–d_1913 | 2016-03-28 → 04-24 | tidak ada |
| easter_2014 | special | d_1164 | d_1137–d_1164 | d_1165–d_1192 | 2014-04-07 → 05-04 | Easter (= Orthodox Easter), Pesach End |
| easter_2015 | special | d_1514 | d_1487–d_1514 | d_1515–d_1542 | 2015-03-23 → 04-19 | Easter, Orthodox Easter, Pesach End |
| year_end_2015 | special | d_1773 | d_1746–d_1773 | d_1774–d_1801 | 2015-12-07 → 2016-01-03 | Chanukah End, **Natal (tutup)**, Tahun Baru |
| holdout | holdout | d_1913 | d_1886–d_1913 | d_1914–d_1941 | 2016-04-25 → 05-22 | Orthodox Easter, Pesach End, Cinco de Mayo, Mother's Day |

- **cv1–cv4** menutup d_1802–d_1913 tanpa celah dan tanpa tumpang tindih. Metriknya
  dirata-rata dan **sebarannya** dilaporkan.
- **Fold khusus** masing-masing menjawab satu pertanyaan dan **tidak pernah dirata-rata**
  ke skor utama. Ada dua fold Easter, supaya ablation Easter tidak bertumpu pada satu
  kejadian. Di 2014 Easter jatuh bersamaan dengan Orthodox Easter, jadi fold itu
  mencampur kedua efek.
- **Holdout** dipakai sekali, di akhir.
- Jendela early stopping tiap fold adalah periode sebelumnya. Itu diperbolehkan karena
  letaknya sebelum cutoff.

## Aturan yang dijaga kode (`validate_folds`, melempar error jika dilanggar)

1. Tepat satu holdout, berakhir di d_1941.
2. Horizon fold lain tidak boleh mencapai holdout.
3. Minimal 4 fold cv, dengan horizon yang tidak tumpang tindih.
4. Minimal 420 hari riwayat sebelum jendela early stopping tiap fold (lag 364 +
   rolling 56).

## Hari tutup

`config.yaml → closed_days` = 25 Desember (Fase 2). `closed_day_mask` menandai tepat
lima hari Natal 2011–2015. Tidak ada yang jatuh di horizon cv maupun holdout; satu jatuh
di `year_end_2015`. Hari tutup dikeluarkan dari baris fitting dan early stopping,
forecast-nya diisi 0, dan tidak ikut dihitung di metrik.

## Pengecekan anti-leakage

`leaking_columns(fn, df, cutoff)`: setiap nilai target setelah cutoff diganti nilai lain;
fungsi fitur dijalankan pada kedua versi; fitur yang berubah di baris d ≤ cutoff + 28
dilaporkan sebagai bocor. Harga dan kalender tidak diacak, karena harga masa depan sudah
tersedia di M5.

Pengecekan ini sendiri diuji pada data asli (30 item CA_1, cutoff d_1857):

| Fitur | Seharusnya | Hasil |
|---|---|---|
| lag 28 / 35 / 364, rata-rata rolling dari lag 28 | aman | lolos |
| rata-rata item dari baris ≤ cutoff | aman | lolos |
| harga relatif terhadap rata-rata item (harga masa depan) | aman | lolos |
| lag 7 | bocor | tertangkap |
| rata-rata item dari semua baris | bocor | tertangkap |
| rata-rata rolling berpusat | bocor | tertangkap |
| lag 28 dengan horizon 29 hari | bocor | tertangkap |

Setiap fungsi fitur di Fase 4 wajib menerima `cutoff` dan punya tes sendiri dengan
pengecekan ini.

## Terbuka untuk Fase 5

Setelah early stopping menentukan jumlah iterasi, pilih salah satu: pertahankan model yang
dilatih pada d < awal early stopping, atau latih ulang pada semua hari ≤ cutoff dengan
jumlah iterasi itu. Pilih satu, terapkan ke semua fold, dan catat di laporan run.
