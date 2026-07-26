# HMDA 2022 — Knowledge Discovery Pipeline

[![Dataset](https://img.shields.io/badge/dataset-FFIEC%20%2F%20CFPB%202022-1f6feb)](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2022)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/dashboard-Dash%20%2F%20Plotly-0f9d78)](https://dash.plotly.com/)

Pipeline penemuan pengetahuan (KDD) lengkap atas sampel **100.000 aplikasi kredit rumah
HMDA 2022**: prapemrosesan, seleksi fitur, empat algoritma clustering, association rule
mining dengan uji signifikansi, deteksi anomali multi-metode, dan dashboard interaktif
lima fase berbahasa Indonesia.

- **Kode sumber:** https://github.com/leonardo-alexander/HMDA
- **Sumber data:** [FFIEC / CFPB — Snapshot National Loan-Level Dataset 2022](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2022)

---

## Temuan utama

| Metrik | Nilai |
|---|---|
| Baris mentah → bersih | 100.000 → 99.995 (5 duplikat) |
| Aplikasi berkeputusan | 67.827 (76,9% Originated / 23,1% Denied) |
| Segmen ditemukan | 7 (K-Means, K dipilih via silhouette) |
| Aturan asosiasi final | 11 dari 28 kandidat |
| Anomali keyakinan tinggi | 739 rekaman (≥3 dari 5 detektor) |

**Pola terkuat:** debt-to-income adalah pemisah dominan. Persetujuan anjlok pada DTI >60%,
dan menambahkan atribut demografis ke aturan DTI inti nyaris tidak mengubah lift-nya.

**Catatan keadilan:** selisih persetujuan antar tract minoritas tidak menutup setelah
stratifikasi DTI (terbesar 12,1 poin persentase). Ini **asosiasi yang perlu diselidiki,
bukan bukti sebab-akibat** — data publik HMDA tidak memuat credit score maupun cadangan dana.

---

## Struktur proyek

```
pipeline/                     Logika Fase 1-5 yang dapat diimpor (satu sumber kebenaran)
  config.py                     konstanta bersama: peran kolom, label, sentinel,
                                spesifikasi binning, penjaga leakage, ambang
  phase1_preprocessing.py       muat, bersihkan, harmonisasi sentinel, dedupe,
                                imputasi, binning, skor & seleksi fitur
  phase2_clustering.py          flag rekayasa, K-Means/DBSCAN/Ward/CLARANS,
                                profiling + penamaan cluster
  phase3_association_rules.py   Apriori, ekstraksi aturan, uji signifikansi,
                                pruning improvement-filter, crosstab geografi × DTI
  phase4_anomaly_detection.py   IQR/Z-score, Isolation Forest, LOF, voting ensemble,
                                triase berbasis bukti
  phase5_reporting.py           ekspor agregat dashboard, dashboard HTML mandiri,
                                komparasi clustering, generator laporan

notebooks/
  HMDA.ipynb                  Deliverable utama. Seluruh narasi (Indonesia), plot, dan
                              sel validasi/assert ada di sini; kode transformasi data
                              diimpor dari pipeline/.

app/
  index.py                    Dashboard Dash 5 fase (Fase 1 Prapemrosesan → Fase 5
                              Pelaporan & Keadilan). Entry point deployment Vercel.
  build_data.py               Membangun ulang setiap CSV yang dibaca dashboard dengan
                              menjalankan pipeline Fase 1-5.

data/
  raw/                        Salinan CSV sumber (reproduksibilitas offline)
  interim/                    Keluaran Fase 1: hmda_clean, hmda_approve_deny, hmda_denials
  processed/                  Keluaran Fase 2-4 + seluruh agregat dash_*.csv

scripts/
  loadtest.py                 Uji beban dashboard: latency, throughput, error.
                              Jalankan app dulu, lalu:
                              python scripts/loadtest.py http://127.0.0.1:8051

results/
  figures/                    Seluruh plot PNG Fase 1-4
  tables/                     Audit seleksi fitur, signifikansi aturan, audit rubric

reports/
  Preprocessing_Report.md            Laporan yang disubmit (tulisan tangan)
  Knowledge_Discovery_Report.md      Laporan yang disubmit (tulisan tangan)
  Metodologi_dan_Justifikasi.md      Justifikasi setiap keputusan analitis
  HMDA_Interactive_Dashboard.html    Dashboard mandiri (generated, tidak di-track git)
```

---

## Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

---

## Menjalankan

### 1. Notebook

Buka `notebooks/HMDA.ipynb` lalu **Restart & Run All**. Sel pertama menambahkan root
proyek ke `sys.path` sehingga `from pipeline import ...` selalu resolve.

Data sumber diambil dari Hugging Face pada run pertama; bila jaringan gagal, pipeline
otomatis fallback ke salinan lokal `data/raw/hmda_sample.csv`.

### 2. Dashboard

```bash
python app/index.py
```

Lalu buka http://127.0.0.1:8050.

| Variabel lingkungan | Fungsi | Default |
|---|---|---|
| `HMDA_DATA_DIR` | Lokasi CSV | `data/processed` |
| `HMDA_PORT` | Port server | `8050` |
| `HMDA_HOST` | Host | `127.0.0.1` |
| `HMDA_DEBUG` | Mode debug (`1` untuk aktif) | `0` |

### 3. Membangun ulang data

```bash
python app/build_data.py
```

Menjalankan seluruh pipeline Fase 1-5 dan menulis ulang setiap CSV yang dibutuhkan
dashboard. Jalankan ini setiap kali logika di `pipeline/` berubah.

---

## Dashboard: lima tab fase KDD

| Tab | Isi |
|---|---|
| **Fase 1 — Prapemrosesan** | Ringkasan pembersihan, segmentasi tipe fitur, missingness per kolom, seleksi fitur, transformasi & penskalaan |
| **Fase 2 — Segmentasi** | Perbandingan K-Means/DBSCAN/Ward/CLARANS dengan metrik validitas, profil 7 segmen, scatter interaktif |
| **Fase 3 — Aturan Asosiasi** | 11 aturan relevan bisnis, lanskap aturan, jaringan aturan, seluruh kandidat mentah |
| **Fase 4 — Deteksi Anomali** | Taksonomi outlier (global/kontekstual/kolektif), skor anomali, triase 15 rekaman ekstrem |
| **Fase 5 — Pelaporan & Keadilan** | Ringkasan eksekutif, geografi, What-If, analisis keadilan |

Setiap fase memuat dropdown **"Mengapa metode ini dipilih?"** yang menjelaskan alasan di
balik tiap ambang dan algoritma.

---

## Metodologi singkat

Justifikasi lengkap ada di [`reports/Metodologi_dan_Justifikasi.md`](reports/Metodologi_dan_Justifikasi.md).

**Prapemrosesan.** Kolom dengan >60% nilai kosong dibuang (missingness struktural); fitur
kontinu diimputasi **median** (tahan skew & outlier, auditable — KNN dihindari karena mahal,
sulit dijelaskan ke regulator, dan berisiko menanamkan korelasi buatan yang lalu "ditemukan"
di Fase 3); kategorikal diisi `"Unknown"` karena kekosongan bermakna. Fitur pasca-keputusan
(interest_rate, total_loan_costs) dikecualikan untuk mencegah **leakage**.

**Audit multikolinearitas.** Cek korelasi berpasangan dilengkapi **VIF** pada 17 fitur
numerik; ambang VIF > 10 setara R² > 0,90 dan **tidak ada fitur yang melewatinya**. Karena
DTI sudah dibinning sehingga luput dari tabel VIF, ditambahkan uji terarah: DTI dipetakan
ke titik tengah band lalu direkonstruksi dari income, besar loan, nilai properti, dan rasio
loan-terhadap-income. Hasilnya **R² = 0,100** (setara VIF 1,11), jadi DTI **tidak** bisa
direkonstruksi dari fitur ukuran dan keduanya dipertahankan.

**Penskalaan berbeda per tujuan.** Clustering memakai winsorize 1/99% + `StandardScaler`
(jarak Euclidean butuh fitur setara). Deteksi anomali memakai `RobustScaler` — median dan
IQR tidak terdistorsi oleh outlier yang justru sedang dicari.

**Perbandingan clustering** (diukur pada matriks yang benar-benar dilihat algoritma):

| Metode | Cakupan | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI vs K-Means |
|---|---|---|---|---|---|
| **K-Means** | Seluruh 99.995 | **0,303** | 1,146 | 21.867 | 1,000 |
| DBSCAN | Sampel 20.000 | 0,297 | 1,090 | 1.810 | 0,849 |
| Hierarchical (Ward) | Sampel 4.000 | 0,294 | 1,152 | 837 | 0,906 |
| CLARANS (k-medoids) | Sampel 4.000 | 0,245 | 1,242 | 768 | 0,710 |
| *K-Means (sampel 4.000)* | Sampel 4.000 | **0,303** | **1,137** | **869** | 1,000 |

K-Means dipakai sebagai segmentasi utama karena unggul pada **ketiga** metrik saat diuji
pada sampel setara, sekaligus satu-satunya yang skalabel ke seluruh data. Ward menyepakatinya
kuat (ARI 0,906), menegaskan struktur 7 segmen bukan artefak satu algoritma. Hierarchical
wajib disampel karena matriks linkage-nya O(n²) memori.

**Association rules.** `min_support` 2% (≈1.357 aplikasi), `lift` > 1,2 (20% di atas
kebetulan), `confidence` ≥ 55% (base rate Denied hanya 23,1%), panjang itemset ≤ 3.
Improvement filter memangkas 28 kandidat menjadi 11 dengan mensyaratkan setiap aturan
mengungguli sub-rule terbaiknya minimal 2 poin.

**Deteksi anomali.** Lima detektor lintas dua filosofi (global: IQR/Z-score/Isolation Forest;
kontekstual: LOF/DBSCAN-noise), konsensus ≥3 suara. 15 rekaman teratas ditriase manual —
seluruhnya berakhir **RARE BUT VALID**, membuktikan nilai ekstrem ≠ kesalahan data.

---

## Deployment

Dashboard di-deploy ke Vercel via `vercel.json` dengan entry point `app/index.py`.
`.vercelignore` mengecualikan notebook, data mentah, dan tabel besar agar bundle tetap
di bawah batas 225 MB.

---

## Catatan

- `RANDOM_STATE = 42` di seluruh pipeline untuk reproduksibilitas.
- Seluruh logika transformasi data ada di `pipeline/`; notebook mengimpor dan memanggilnya.
  Narasi, plot, dan sel validasi tetap inline di notebook.
- Sel audit rubric sengaja tidak dipindahkan ke `pipeline/` karena melakukan introspeksi
  `globals()` notebook — sifatnya memang notebook-only.
- Temuan keadilan adalah **asosiasi**, bukan klaim kausal (lihat catatan di atas).
