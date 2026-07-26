# Key Insights & Recommendations
### HMDA 2022 Knowledge Discovery — Kelompok 4

Seluruh angka di dokumen ini berasal dari keluaran notebook dan file CSV hasil ekspor.
Temuan dinyatakan sebagai **asosiasi historis**, bukan hubungan sebab-akibat: data publik HMDA
tidak memuat credit score, cadangan dana, riwayat pembayaran, maupun kebijakan internal lender.

---

## A. LIMA TEMUAN UTAMA

### 1. Beban utang mendominasi keputusan, tetapi jarang dinyatakan sebagai alasannya

**Finding:**
DTI adalah pemisah terkuat dalam seluruh proses mining, namun hanya sebagian kecil penolakan
yang secara resmi mencantumkan debt-to-income sebagai alasan. Pola yang paling menentukan
justru yang paling jarang dilaporkan.

**Evidence:**
- **Fase 3:** `debt_to_income_ratio=>60% → Denied`, confidence **91,5%**, lift **3,96**, n=4.121.
  Ditambah `DTI>60% + Subordinate_Lien → Denied`, confidence **94,0%**, lift **4,06**, n=1.555.
- **Fase 3 (alasan penolakan):** hanya **8,7%** penolakan mencantumkan "Debt-to-income";
  **72,9%** masuk kategori "Other".
- **Fase 2:** segmen *DTI-stressed borrowers* (8.731 aplikasi, 8,7% portfolio) punya tingkat
  persetujuan **39,5%**, terendah kedua setelah manufactured housing.

**Why it matters:**
Faktor yang paling kuat berasosiasi dengan penolakan hampir tidak terlihat di data alasan resmi,
sehingga pelaporan internal bisa salah menggambarkan penyebab penolakan.

**Recommendation:**
Tambahkan pemeriksaan DTI di tahap intake sebelum underwriting penuh, dan perbaiki taksonomi
alasan penolakan agar kategori "Other" yang mencapai 72,9% dapat diuraikan.

**Confidence / caveat:**
Ini asosiasi kuat dan konsisten lintas fase, tetapi tingginya proporsi "Other" berarti alasan
resmi tidak dapat dipakai untuk memastikan mekanismenya.

---

### 2. Manufactured housing adalah penalti struktural, bukan sekadar efek income rendah

**Finding:**
Aplikasi manufactured housing membentuk segmen yang paling jelas terpisah di seluruh analisis,
dengan tingkat persetujuan jauh di bawah portfolio. Penaltinya melekat pada jenis properti dan
jalur pembiayaan, bukan semata pada income pemohon.

**Evidence:**
- **Fase 2:** segmen *Manufactured-housing applicants* (4.529 aplikasi) hanya **43,1%** disetujui,
  sementara portfolio **76,9%**.
- **Fase 2 (validitas):** segmen ini paling kohesif dari tujuh segmen, silhouette **0,495**,
  dengan Ward purity **1,000** dan CLARANS purity **1,000** — ketiga algoritma menemukan
  kelompok yang sama.
- **Fase 3:** `construction_method=Manufactured → Denied` confidence **56,9%** lift **2,46**;
  dikombinasikan dengan `loan_type=Conventional` naik ke confidence **63,5%** lift **2,75**.
- **Fase 4:** teridentifikasi sebagai kandidat kolektif dengan skor bukti **4/4**, ditandai
  sebagai segmen langka dan terisolasi yang stabil, dengan hanya **8,7%** anggotanya pernah
  ditandai detektor individual.

**Why it matters:**
Segmen ini ditolak melalui jalur konvensional pada tingkat yang jauh lebih tinggi, padahal
produk pembiayaan yang sesuai untuk jenis properti ini memang berbeda.

**Recommendation:**
Sediakan jalur produk khusus seperti chattel lending atau FHA Title I untuk manufactured
housing, alih-alih memaksanya melalui underwriting konvensional.

**Confidence / caveat:**
Didukung tiga fase sekaligus, tetapi HMDA tidak memuat kondisi properti maupun status
kepemilikan tanah yang ikut menentukan kelayakan agunan.

---

### 3. Selisih persetujuan antar kelompok justru paling lebar di kelompok berisiko rendah

**Finding:**
Setelah distratifikasi menurut beban utang, selisih persetujuan antar lingkungan dan antar
kelompok gender tidak menutup. Polanya berlawanan dengan dugaan umum: selisih terbesar berada
di kelompok DTI rendah, yaitu kelompok yang paling aman secara finansial.

**Evidence:**
- **Fase 5 (tract minority × DTI):** pada DTI rendah (<36%), tract minoritas-rendah **85,5%**
  versus tract mayoritas-minoritas **73,4%**, selisih **12,1 poin**. Pada DTI tinggi (>50%)
  selisihnya menyempit menjadi **1,8 poin**.
- **Fase 5 (gender × DTI):** pada DTI rendah, aplikasi Joint **88,6%** versus Female **80,9%**
  dan Male **80,0%**. Secara keseluruhan Joint **83,0%**, Male **73,9%**, Female **72,3%**.
- **Fase 3:** menambahkan atribut demografis ke aturan DTI inti tidak lolos improvement filter,
  artinya demografi tidak menambah daya pisah di atas DTI.

**Why it matters:**
Selisih yang bertahan justru di kelompok paling aman adalah sinyal yang perlu ditelaah lebih
lanjut untuk kepatuhan fair lending.

**Recommendation:**
Jadikan segmen DTI rendah sebagai prioritas tinjauan fair lending internal, memakai data
underwriting lengkap yang tidak tersedia di HMDA publik.

**Confidence / caveat:**
Ini disparitas terukur, bukan bukti diskriminasi: credit score, cadangan dana, dan
loan-to-value pemohon tidak tersedia sehingga faktor perancu tidak dapat dikesampingkan.

---

### 4. Nilai ekstrem dalam data ini terbukti sah, bukan kesalahan data

**Finding:**
Tidak satu pun rekaman paling ekstrem terbukti sebagai kesalahan data. Kedua filosofi deteksi
juga terbukti menangkap kelompok rekaman yang berbeda, sehingga memakai satu pendekatan saja
akan melewatkan seluruh kelas anomali yang lain.

**Evidence:**
- **Fase 4:** dari lima detektor, **739** aplikasi memperoleh minimal tiga suara. 15 rekaman
  paling ekstrem ditriase manual dan **seluruhnya** berakhir **RARE BUT VALID**, nol kesalahan data.
- **Fase 4 (taksonomi):** **9.959** outlier global (10,0%), **589** kontekstual/lokal (0,6%),
  dan hanya **476** tertangkap keduanya, menunjukkan kedua pendekatan tidak saling menggantikan.
- **Fase 4 (kolektif tingkat grup):** dari 4 kelompok non-geografis yang ditandai Isolation
  Forest, **3 di antaranya manufactured housing**, memperkuat temuan nomor 2 lewat jalur
  bukti yang berbeda.

**Why it matters:**
Memakai ensemble anomali sebagai aturan hapus otomatis akan membuang aplikasi jumbo yang sah.

**Recommendation:**
Perlakukan keluaran deteksi anomali sebagai antrean tinjauan manual, bukan filter penghapusan.

**Confidence / caveat:**
Triase manual baru mencakup 15 rekaman teratas, sehingga tidak menjamin seluruh 739 anomali
berkeyakinan tinggi juga sah. Sisa **29** kelompok kolektif yang ditandai berbasis negara bagian
dan **52%** di antaranya yurisdiksi kecil, sehingga sengaja tidak diklaim sebagai temuan.

---

### 5. Segmentasi terbukti stabil, dan tidak ditemukan cluster palsu

**Finding:**
Struktur tujuh segmen bukan artefak satu algoritma. Audit stabilitas tidak menemukan satu pun
cluster yang mencurigakan atau terbentuk semu.

**Evidence:**
- **Fase 2 (audit stabilitas):** ketujuh segmen memiliki `spurious_cluster_candidate = False`.
- **Fase 2 (perbandingan metode):** pada sampel 4.000 baris yang sama, K-Means unggul di ketiga
  metrik, silhouette **0,303**, Davies-Bouldin **1,137**, Calinski-Harabasz **869**.
- **Fase 2 (replikasi):** Ward menyepakati K-Means dengan ARI **0,906**; CLARANS **0,710**,
  perbedaan yang wajar karena mengoptimalkan medoid, bukan centroid.
- **Catatan jujur:** satu segmen paling lemah replikasinya, yaitu *Jumbo / high-net-worth buyers*
  dengan Ward purity **0,584** dan CLARANS purity **0,345**, sehingga batasnya paling kabur.

**Why it matters:**
Segmentasi cukup stabil untuk dijadikan dasar kebijakan, kecuali segmen jumbo yang perlu
diperlakukan lebih hati-hati.

**Recommendation:**
Pakai enam segmen yang replikasinya kuat sebagai dasar penyusunan kebijakan produk, dan tinjau
ulang definisi segmen jumbo sebelum memakainya untuk keputusan operasional.

**Confidence / caveat:**
Validitas diukur pada sampel 4.000 baris untuk Ward dan CLARANS karena keterbatasan memori,
bukan pada seluruh populasi.

---

## B. KESIMPULAN KESELURUHAN

Proses mining ini mengungkap hal yang tidak terlihat dari membaca rekaman HMDA satu per satu.
Pertama, keputusan kredit dalam data ini terorganisir di sekitar **kapasitas membayar**, bukan
ukuran pinjaman: DTI di atas 60% berasosiasi dengan penolakan pada confidence 91,5% dan lift
3,96, sementara segmen jumbo dengan pinjaman terbesar justru disetujui 87,4%. Kedua, terdapat
penalti yang melekat pada **jenis properti**: manufactured housing muncul sebagai segmen paling
kohesif sekaligus paling ditolak, dan ketiga fase berbeda menunjuk kelompok yang sama. Ketiga,
**alasan penolakan resmi tidak mencerminkan pola yang sebenarnya**, karena 72,9% masuk kategori
"Other" sementara DTI hanya 8,7%. Keempat, nilai ekstrem ternyata **sah secara aritmetika**, dan
dan pola kolektif tingkat kelompok justru menunjuk balik ke manufactured housing. Kelima, selisih
persetujuan antar kelompok demografis dan geografis **bertahan setelah beban utang disamakan**,
dan paling lebar justru pada kelompok berisiko rendah. Gabungan inilah yang tidak tampak dari
data mentah: struktur keputusan, penalti produk, keterbatasan pelaporan, dan disparitas residual
yang menuntut penyelidikan lanjutan.

---

## C. REKOMENDASI EKSEKUTIF

### 1. Pindahkan penyaringan DTI ke tahap intake  *(prioritas tertinggi)*

- **Aksi:** Terapkan pemeriksaan DTI sebelum underwriting penuh, dengan rujukan alternatif untuk
  pemohon di atas 60%.
- **Bukti:** DTI>60% berasosiasi dengan penolakan pada confidence 91,5%, lift 3,96, mencakup
  4.121 aplikasi; segmen DTI-stressed hanya 39,5% disetujui.
- **Nilai bisnis:** Memangkas biaya underwriting pada berkas yang secara historis hampir pasti
  ditolak, dan mempercepat keputusan bagi pemohon.
- **Kehati-hatian:** Ini pola historis, bukan aturan kelayakan. Jangan dijadikan penolakan
  otomatis tanpa tinjauan manusia.

### 2. Bangun jalur produk khusus manufactured housing

- **Aksi:** Sediakan atau bermitra pada program chattel lending atau FHA Title I.
- **Bukti:** Segmen 4.529 aplikasi dengan persetujuan 43,1%; aturan Manufactured+Conventional →
  Denied confidence 63,5% lift 2,75; segmen paling kohesif dengan silhouette 0,495 dan purity 1,000.
- **Nilai bisnis:** Membuka segmen yang saat ini gagal melalui jalur konvensional, sekaligus
  menurunkan biaya underwriting yang berakhir ditolak.
- **Kehati-hatian:** Jenis properti berkorelasi dengan income dan geografi, sehingga eksposur
  fair lending perlu dipantau.

### 3. Prioritaskan tinjauan fair lending pada segmen DTI rendah

- **Aksi:** Audit internal atas selisih persetujuan di kelompok DTI rendah memakai data
  underwriting lengkap.
- **Bukti:** Selisih 12,1 poin antar tract minoritas dan 7,7 poin antara Joint dan Female, keduanya
  pada DTI rendah, justru mengecil pada DTI tinggi.
- **Nilai bisnis:** Menurunkan risiko kepatuhan sebelum menjadi temuan regulator.
- **Kehati-hatian:** Data publik tidak memuat credit score dan cadangan dana, jadi ini sinyal
  penyelidikan, bukan kesimpulan diskriminasi.

### 4. Perbaiki taksonomi alasan penolakan

- **Aksi:** Uraikan kategori "Other" menjadi alasan yang dapat ditindaklanjuti.
- **Bukti:** 72,9% penolakan tercatat sebagai "Other", sementara DTI yang terbukti paling kuat
  hanya 8,7%.
- **Nilai bisnis:** Tanpa ini, pelaporan internal tidak dapat menjelaskan penolakannya sendiri.
- **Kehati-hatian:** Kategori "Other" sebagian ditentukan format pelaporan HMDA, sehingga tidak
  seluruhnya dapat dikendalikan lender.

### 5. Jadikan deteksi anomali sebagai antrean tinjauan, bukan filter hapus

- **Aksi:** Arahkan keluaran ensemble ke tinjauan manual dengan uji konsistensi internal.
- **Bukti:** 739 aplikasi mendapat ≥3 suara detektor, tetapi 15 teratas seluruhnya terbukti
  RARE BUT VALID tanpa satu pun kesalahan data.
- **Nilai bisnis:** Mencegah terbuangnya aplikasi jumbo yang sah.
- **Kehati-hatian:** Baru 15 rekaman yang ditriase manual, sehingga sisanya belum tervalidasi.

---

*Sumber: `data/processed/` dan `results/tables/`, dihasilkan oleh notebook HMDA.ipynb dan
`python app/build_data.py`.*
