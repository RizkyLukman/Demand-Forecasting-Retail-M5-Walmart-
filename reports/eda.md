# Fase 2 — EDA terarah

Sumber: `python -m retailfc.eda --config config.yaml` (angka: `reports/eda.json`).
Data: tabel panjang dari Fase 1 (baris sejak penjualan pertama tiap item-toko),
**hanya d_1 … d_1913**. Holdout (d_1914–d_1941) tidak dipakai.

Catatan: horizon fold CV (d_1802–d_1913) ada di dalam jendela ini. EDA dipakai untuk
merancang fitur, bukan untuk memilih model, jadi hal ini diterima dan dicatat di sini.

---

## P1. Proporsi hari tanpa penjualan per kategori

| Kategori | Proporsi nol | Rentang di 10 toko |
|---|---:|---|
| FOODS | 51,1% | 40,9% (CA_3) – 56,8% (TX_1) |
| HOUSEHOLD | 64,6% | 54,0% (CA_3) – 72,0% (CA_4) |
| HOBBIES | 71,5% | 63,4% (CA_3) – 78,5% (WI_2) |
| **Semua** | **59,6%** | |

Bahkan setelah hari sebelum penjualan pertama dibuang, sebagian besar hari item bernilai
nol. HOBBIES paling intermiten. CA_3 (toko terbesar) punya proporsi nol terendah di setiap
kategori.

**Implikasi:** model bisa terlihat bagus hanya dengan menebak mendekati nol. Fase 5/6
wajib melaporkan metrik per segmen dan mengecek bias, sesuai rencana.

## P2a. Pola mingguan

Unit per hari dalam minggu relatif terhadap rata-rata mingguan, setelah setiap hari
dibagi rata-rata 28 hari berpusat (menghilangkan tren dan musim tahunan). Hari tutup
tidak dihitung.

![profil hari](figures/eda_weekday.png)

| | Sen | Sel | Rab | Kam | Jum | Sab | Min | maks/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOODS | 0,96 | 0,89 | 0,88 | 0,88 | 0,99 | 1,19 | 1,20 | 1,37 |
| HOBBIES | 0,93 | 0,89 | 0,90 | 0,90 | 1,04 | 1,23 | 1,11 | 1,38 |
| HOUSEHOLD | 0,94 | 0,86 | 0,85 | 0,86 | 1,01 | 1,26 | 1,22 | 1,48 |

Kuat dan konsisten: akhir pekan terjual 37–48% lebih banyak daripada hari terlemah
(Selasa–Kamis). HOBBIES memuncak di hari Sabtu lalu turun lagi di hari Minggu; dua
kategori lain tetap tinggi di hari Minggu.

**Implikasi:** hari dalam minggu wajib jadi fitur. Lag kelipatan 7 (28, 35, 42, 49, 56)
menjaga hari tetap sejajar.

## P2b. Efek SNAP per negara bagian

Hari SNAP dari `calendar.csv`:
CA = tanggal 1–10; TX = 1, 3, 5, 6, 7, 9, 11, 12, 13, 15; WI = 2, 3, 5, 6, 8, 9,
11, 12, 14, 15.

Kenaikan = unit di hari SNAP vs hari non-SNAP, dibandingkan **pada hari yang sama dalam
minggu** lalu dirata-rata (sudah dihilangkan trennya seperti di P2a). Bantuan SNAP hanya
bisa dipakai membeli makanan, jadi HOBBIES dan HOUSEHOLD berfungsi sebagai **placebo**:
kenaikan di keduanya mengukur efek umum awal bulan (mis. gajian) yang bercampur dengan
hari SNAP.

| Negara bagian | FOODS | HOUSEHOLD (placebo) | HOBBIES (placebo) | FOODS dikurangi placebo |
|---|---:|---:|---:|---:|
| CA | +10,2% | +3,8% | +2,7% | ~+7 poin |
| TX | +14,9% | +2,6% | +1,9% | ~+12 poin |
| WI | +28,1% | +3,0% | +1,3% | ~+25 poin |

Kenaikan FOODS jauh di atas placebo di semua negara bagian, jadi sebagian besarnya adalah
efek SNAP yang nyata, paling kuat di WI. Placebo menunjukkan efek awal bulan yang kecil
(~2–4%).

**Implikasi:** pakai `snap_<negara bagian toko>` sebagai fitur, terutama berguna untuk
FOODS. Ini asosiasi dari data observasi, bukan estimasi kausal.

## P3. Hari toko tutup

**25 Desember praktis hari tutup, tetapi tidak persis nol.** Diverifikasi dari data:

| Tanggal | Unit, seluruh jaringan (10 toko) |
|---|---:|
| 2011-12-25 | 13 |
| 2012-12-25 | 11 |
| 2013-12-25 | 20 |
| 2014-12-25 | 20 |
| 2015-12-25 | 14 |

- Tidak ada hari dengan penjualan nol di **semua** toko, jadi aturan "total = 0" saja
  akan melewatkan Natal.
- Setiap Natal berada di bawah 10% rata-rata 28 hari berpusatnya. **Tidak ada hari lain
  yang begitu.**
- Hari dengan total nol di level toko hanya terjadi pada 25 Desember (mis. CA_1 dan CA_4
  kelima tahun, CA_3 hanya 2014).

**Keputusan untuk fase berikutnya:** perlakukan 25 Desember sebagai hari tutup.
Keluarkan dari target training dan statistik rolling, dan isi forecast-nya dengan 0.
Natal tidak ada di horizon fold CV mana pun maupun holdout (yang terakhir d_1792,
2015-12-25), jadi ini hanya berpengaruh ke training.

## P4. Penjualan di sekitar Easter (persiapan P4)

Unit per kategori di setiap hari sekitar Minggu Easter, dibagi rata-rata **hari yang
sama** 2–5 minggu sebelum dan sesudahnya (jendela event tidak ikut di baseline).
Tahun 2011–2015. Easter 2016 (d_1885) tidak dipakai karena baseline-nya butuh hari
setelah d_1913.

![jendela easter](figures/eda_easter.png)

| Hari dari Easter | −3 (Kam) | −2 (Jum) | −1 (Sab) | 0 (Min) | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FOODS | 1,04 | **1,16** | **1,15** | 0,91 | 0,96 | 0,99 | 1,02 |
| HOBBIES | 0,93 | 0,99 | 0,98 | **0,78** | 1,09 | **1,13** | 1,12 |
| HOUSEHOLD | 1,00 | 1,09 | 1,00 | **0,83** | 1,04 | 1,04 | 1,06 |

- **FOODS:** +15–16% pada Jumat dan Sabtu sebelum Easter, −9% di Minggu Easter.
- **HOBBIES dan HOUSEHOLD:** turun di Minggu Easter (−22% / −17%). HOBBIES lalu terjual
  9–13% lebih banyak di hari +1 sampai +3. Penyebabnya tidak terlihat dari data.
- Efeknya terkumpul di jendela pendek, sekitar **−3 sampai +3 hari**. Antar tahun sangat
  bervariasi (garis tipis), dengan lima kejadian per kategori.
- Kalender hanya menandai Minggu Easter itu sendiri. Flag `is_event` biasa tidak bisa
  menyatakan "dua hari sebelumnya". Ini alasan dibutuhkannya `days_to_event` /
  `days_since_event`.

**Implikasi untuk P4 (Fase 4):** batasi `days_to_easter` / `days_since_easter` di sekitar
±7 hari. Ablation butuh fold yang horizon-nya memuat hari-hari sekitar Easter: CV 3
berakhir di Easter 2016 (d_1885) dan mencakup hari-hari sebelumnya. Fold tambahan di
sekitar Easter 2015 (d_1528) dibutuhkan untuk hari-hari sesudahnya.

## P5. Kualitas data: harga anomali dan deretan nol panjang

Ditambahkan setelah pertanyaan "apakah data ini tidak perlu preprocessing?". Kode:
`q5_data_quality` di `eda.py`; angka di `reports/eda.json → q5_data_quality`. Harga dicek
sampai minggu d_1913, penjualan sampai d_1913.

### Harga anomali: kecil

| Pengecekan | Baris |
|---|---:|
| Baris harga yang dicek | 6.597.201 |
| Harga < 0,10 dolar | 231 (26 item) |
| Harga < 20% median item-toko itu sendiri | **405** |
| Harga > 5× median item-toko itu sendiri | **34** |

Harga absolut yang sangat rendah **belum tentu salah**: contoh terbanyak adalah item FOODS_3
yang memang murah (0,05 dolar, median 0,20). Kandidat anomali yang lebih tepat adalah harga
yang jauh dari median item-toko itu sendiri (405 + 34 baris, 0,007% data).

**Keputusan:** dibiarkan dan didokumentasikan. Dampaknya sangat kecil, dan model pohon
tahan terhadap nilai ekstrem. Fitur `price_rel_max` / `is_discount` akan membaca harga
seperti itu sebagai diskon besar.

### Deretan nol panjang: indikasi kuat stok habis (censored demand)

Panjang deretan hari tanpa penjualan dihitung per deret item-toko (hari tutup tidak
dihitung). "Item laris" = terjual di ≥ 50% hari bukanya.

| Pengecekan | Jumlah |
|---|---:|
| Item laris | 10.004 dari 30.490 deret |
| Item laris dengan deretan nol ≥ 28 hari | **6.235 (62%)** |
| Item laris dengan deretan nol ≥ 56 hari | 4.591 (46%) |
| Item laris dengan deretan nol ≥ 90 hari | 3.117 (31%) |
| Deret yang nol terus di ≥ 28 hari terakhir sebelum d_1913 | 1.587 (240 di antaranya item laris) |
| **Porsi semua hari bernilai nol yang berada di dalam deretan nol ≥ 28 hari** | **29,5%** |
| Porsi yang sama, khusus item laris | 23,9% |

**Analisis:** untuk item yang laku di lebih dari separuh harinya, tidak terjual 28 hari
berturut-turut karena kebetulan hampir mustahil (peluangnya kira-kira 0,5²⁸ jika hari-hari
saling bebas). Deretan seperti ini hampir pasti berarti barang **tidak tersedia di rak**
(stok habis, ditarik, atau diganti), bukan tidak ada pembeli. Artinya sebagian nol di data
adalah *penjualan nol*, bukan *permintaan nol*.

**Dampak:** model belajar dari nol-nol ini, sehingga bisa terlalu rendah memprediksi item
yang sebenarnya diminati. Untuk simulasi stok (Fase 7), yang dibutuhkan adalah permintaan,
bukan penjualan.

**Keputusan:**
1. **Fase 4 (sudah dikerjakan):** fitur `zero_run_lag28` = panjang deretan nol per 28 hari
   lalu. Fitur ini lolos tes anti-leakage. Pengecekan awal di CA_1: rata-rata penjualan hari
   ini 0,44 unit bila `zero_run_lag28` ≥ 28, dibanding 1,78 unit bila tidak.
2. **Fase 5:** uji pembanding, yaitu baris di dalam deretan nol panjang item laris
   dikeluarkan dari training (diperlakukan seperti hari tutup). Dampaknya diukur, bukan
   ditebak. Evaluasi tetap memakai penjualan aktual.
3. **Fase 7:** simulasi stok sebaiknya memakai versi yang mengeluarkan stok habis dari
   training, karena yang disimulasikan adalah permintaan.

---

## Ringkasan keputusan yang diteruskan

| Topik | Keputusan | Fase |
|---|---|---|
| Natal | Anggap tutup: keluarkan dari target/statistik rolling, forecast 0 | 4–5 |
| Hari dalam minggu | Fitur wajib; pertahankan lag kelipatan 7 | 4 |
| SNAP | Fitur `snap` sesuai negara bagian toko | 4 |
| Easter | `days_to/since_easter`, dibatasi ±7; fold tambahan sekitar d_1528 | 3–4 |
| Nol | Laporkan metrik per segmen dan bias; waspadai model yang "menang" karena menebak nol | 5–6 |
| Stok habis | Fitur `zero_run_lag28` (sudah); uji keluarkan deretan nol panjang dari training | 4–5, 7 |
| Harga anomali | Dibiarkan, didokumentasikan (405 + 34 baris) | – |
