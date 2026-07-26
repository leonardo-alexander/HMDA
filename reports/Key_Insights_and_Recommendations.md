# Key Insights & Recommendations
### HMDA 2022 Knowledge Discovery — Kelompok 4

Seluruh angka di dokumen ini berasal dari keluaran notebook dan file CSV hasil ekspor.
Temuan dinyatakan sebagai **asosiasi historis**, bukan hubungan sebab-akibat: data publik HMDA
tidak memuat credit score, cadangan dana, riwayat pembayaran, maupun kebijakan internal lender.

---

## A. LIMA TEMUAN UTAMA

### 1. Beban utang mendominasi keputusan, dan alasan resmi lender menyepakatinya

**Finding:**
DTI adalah pemisah terkuat dalam seluruh proses mining, dan hal itu dikonfirmasi secara
independen oleh alasan penolakan yang dicatat lender sendiri. Dua sumber bukti yang tidak
saling bergantung menunjuk faktor yang sama.

**Evidence:**
- **Fase 3 (association rule):** `debt_to_income_ratio=>60% → Denied`, confidence **91,5%**,
  lift **3,96**, n=4.121. Ditambah `DTI>60% + Subordinate_Lien → Denied`, confidence **94,0%**,
  lift **4,06**, n=1.555.
- **Fase 3 (alasan penolakan):** **28,4%** penolakan mencantumkan "Debt-to-income", yang
  tertinggi dari seluruh kategori. Aturan asosiasi ditambang tanpa melihat field ini sama
  sekali, sehingga kesepakatannya bukan hasil sirkularitas.
- **Fase 2:** segmen *DTI-stressed borrowers* (8.731 aplikasi, 8,7% portfolio) punya tingkat
  persetujuan **39,5%**, terendah kedua setelah manufactured housing.

**Why it matters:**
Faktor pendorong penolakan terbesar sudah teridentifikasi jelas dan dapat diperiksa jauh
sebelum underwriting penuh dijalankan.

**Recommendation:**
Terapkan pemeriksaan DTI di tahap intake, dengan rujukan alternatif bagi pemohon di atas 60%
alih-alih meneruskan berkas ke underwriting penuh.

**Confidence / caveat:**
Kesepakatan dua sumber bukti memperkuat pola ini, tetapi tetap asosiasi historis: alasan
penolakan dicatat setelah keputusan dibuat, sehingga tidak dapat membuktikan arah sebab-akibat.

---

### 1b. Faktor terbesar kedua justru tidak ada dalam data

**Finding:**
"Credit history" adalah alasan penolakan kedua terbanyak, padahal credit score sama sekali
tidak tersedia di HMDA publik. Sebagian besar kekuatan penjelas keputusan berada di luar
jangkauan dataset ini.

**Evidence:**
- **Fase 3 (alasan penolakan):** **25,3%** penolakan mencantumkan "Credit history", disusul
  Collateral **14,3%** dan Other **11,6%**.
- **Fase 1:** tidak satu pun dari 99 kolom mentah memuat credit score, cadangan dana, maupun
  riwayat pembayaran.

**Why it matters:**
Ini menjelaskan mengapa selisih persetujuan antar kelompok tidak bisa langsung dibaca sebagai
diskriminasi: variabel penjelas terbesar kedua memang tidak terlihat.

**Recommendation:**
Setiap analisis fair lending lanjutan wajib memakai data underwriting internal, bukan HMDA
publik saja.

**Confidence / caveat:**
Angka ini berasal dari alasan yang dilaporkan lender, sehingga mencerminkan praktik pencatatan
mereka dan belum tentu bobot sebenarnya dalam keputusan.

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
- **Fase 4:** proporsi anggota yang ditandai detektor individual **8,75%**, sedikit di bawah
  baseline portfolio **11,0%**. Ini melengkapi gambaran, tetapi bukan bukti outlier kolektif:
  dua segmen lain justru lebih rendah lagi, **6,06%** dan **5,32%**.

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
Setelah beban utang disamakan, selisih persetujuan antar lingkungan tetap ada dan justru paling
besar di kelompok DTI rendah, yaitu kelompok yang paling aman secara finansial. Sebaliknya,
selisih gender praktis hilang begitu struktur pemohon disetarakan.

**Evidence:**
- **Fase 5 (tract minority × DTI):** pada DTI rendah (<36%), tract minoritas-rendah **85,5%**
  versus tract mayoritas-minoritas **73,4%**, selisih **12,1 poin**. Pada DTI tinggi (>50%)
  selisihnya menyempit menjadi **1,8 poin**.
- **Fase 5 (gender × DTI):** setelah dibatasi pada **35.191** berkas pemohon tunggal, selisih
  gender nyaris hilang: Male **73,5%** versus Female **71,9%**, cuma **1,6 poin**. Pada DTI
  rendah Female justru sedikit lebih tinggi, **80,8%** berbanding **79,5%**. Kategori `Joint`
  sengaja dikeluarkan karena bukan gender, melainkan penanda berkas dua pemohon: **99,9%**
  di antaranya punya co-applicant dengan median income **$119rb** berbanding Female **$71rb**.
- **Fase 3:** menambahkan atribut demografis ke aturan DTI inti tidak lolos improvement filter,
  artinya demografi tidak menambah daya pisah di atas DTI.

**Why it matters:**
Selisih antar lingkungan yang justru paling besar di kelompok paling aman adalah sinyal yang
perlu ditelaah untuk kepatuhan fair lending. Sebaliknya, gender tidak menunjukkan pola serupa
setelah struktur pemohon disetarakan.

**Recommendation:**
Jadikan segmen DTI rendah sebagai prioritas tinjauan fair lending internal, memakai data
underwriting lengkap yang tidak tersedia di HMDA publik.

**Confidence / caveat:**
Ini disparitas terukur, bukan bukti diskriminasi: credit score, cadangan dana, dan
loan-to-value pemohon tidak tersedia sehingga faktor perancu tidak dapat dikesampingkan.

---

### 4. Nilai ekstrem terbukti sah, dan kelas kolektif tidak terbukti ada

**Finding:**
Tidak satu pun rekaman paling ekstrem terbukti sebagai kesalahan data. Kedua filosofi deteksi
juga terbukti menangkap kelompok rekaman yang berbeda, sehingga memakai satu pendekatan saja
akan melewatkan seluruh kelas anomali yang lain.

**Evidence:**
- **Fase 4:** dari lima detektor, **739** aplikasi memperoleh minimal tiga suara. 15 rekaman
  paling ekstrem ditriase manual dan **seluruhnya** berakhir **RARE BUT VALID**, nol kesalahan data.
- **Fase 4 (taksonomi):** **9.959** outlier global (10,0%), **589** kontekstual/lokal (0,6%),
  dan hanya **476** tertangkap keduanya, menunjukkan kedua pendekatan tidak saling menggantikan.
- **Fase 4 (kolektif tingkat grup): hasil negatif.** Isolation Forest pada profil kelompok
  menandai **33** kelompok, tetapi tidak satu pun memenuhi definisi outlier kolektif. Definisi
  itu menuntut anggota yang normal secara individual, sedangkan **28 dari 33** kelompok justru
  punya proporsi anggota anomali **di atas** baseline portfolio **11,0%**, dengan median
  **19,1%**. Label *pure collective* juga tidak membedakan apa pun, karena ambang di bawah 25%
  terlalu longgar untuk baseline 11,0%.

**Why it matters:**
Memakai ensemble anomali sebagai aturan hapus otomatis akan membuang aplikasi jumbo yang sah.

**Recommendation:**
Perlakukan keluaran deteksi anomali sebagai antrean tinjauan manual, bukan filter penghapusan.

**Confidence / caveat:**
Triase manual baru mencakup 15 rekaman teratas, sehingga tidak menjamin seluruh 739 anomali
berkeyakinan tinggi juga sah. Untuk kelas kolektif, hasil negatif dilaporkan apa adanya: dari 33
kelompok yang ditandai, **15** berasal dari yurisdiksi kecil dan **14** dari negara bagian besar,
sehingga yang tertangkap sebagian besar efek ukuran kelompok dan geografi.

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
**alasan resmi lender menyepakati temuan mining**, karena DTI tercatat 28,4% sebagai alasan
terbanyak, sementara faktor terbesar kedua yaitu credit history 25,3% justru tidak ada di data. Keempat, nilai ekstrem ternyata **sah secara aritmetika**, dan
sementara kelas outlier kolektif diperiksa penuh dan hasilnya negatif. Kelima, selisih
persetujuan antar lingkungan **tetap ada setelah beban utang disamakan** dan paling lebar justru
pada kelompok berisiko rendah, sementara selisih gender justru hilang setelah struktur pemohon
disetarakan. Gabungan inilah yang tidak tampak dari
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
- **Bukti:** Selisih 12,1 poin antar tract minoritas pada DTI rendah, mengecil jadi 1,8 poin pada
  DTI tinggi. Selisih gender tidak dimasukkan karena hanya 1,6 poin setelah dibatasi pemohon tunggal.
- **Nilai bisnis:** Menurunkan risiko kepatuhan sebelum menjadi temuan regulator.
- **Kehati-hatian:** Data publik tidak memuat credit score dan cadangan dana, jadi ini sinyal
  penyelidikan, bukan kesimpulan diskriminasi.

### 4. Lengkapi analisis dengan data credit history

- **Aksi:** Gabungkan data credit score internal sebelum menarik kesimpulan apa pun soal disparitas.
- **Bukti:** Credit history adalah alasan penolakan terbanyak kedua di 25,3%, tetapi tidak ada
  satu pun kolom HMDA yang memuatnya.
- **Nilai bisnis:** Mencegah kesimpulan fair lending yang keliru karena variabel penjelas utama hilang.
- **Kehati-hatian:** Selama variabel ini absen, seluruh selisih yang terlihat tetap berstatus
  sinyal untuk diselidiki, bukan temuan final.

### 5. Jadikan deteksi anomali sebagai antrean tinjauan, bukan filter hapus

- **Aksi:** Arahkan keluaran ensemble ke tinjauan manual dengan uji konsistensi internal.
- **Bukti:** 739 aplikasi mendapat ≥3 suara detektor, tetapi 15 teratas seluruhnya terbukti
  RARE BUT VALID tanpa satu pun kesalahan data.
- **Nilai bisnis:** Mencegah terbuangnya aplikasi jumbo yang sah.
- **Kehati-hatian:** Baru 15 rekaman yang ditriase manual, sehingga sisanya belum tervalidasi.

---

*Sumber: `data/processed/` dan `results/tables/`, dihasilkan oleh notebook HMDA.ipynb dan
`python app/build_data.py`.*
