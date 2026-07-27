# Metodologi dan Justifikasi Keputusan Analitis
### Pipeline Penemuan Pengetahuan HMDA 2022 — Kelompok 3

Dokumen ini menjawab satu pertanyaan untuk setiap keputusan dalam pipeline: **mengapa A dan bukan B?**
Setiap angka di sini diambil langsung dari hasil eksekusi pipeline, bukan perkiraan.

**Ringkasan data:** 100.000 baris mentah → 99.995 baris bersih (5 duplikat dihapus) → 67.827 aplikasi
berkeputusan (Originated vs Denied) → base rate 76,9% Originated / 23,1% Denied.

---

## Fase 1 — Prapemrosesan

### 1.1 Mengapa kolom dengan >60% nilai kosong dibuang?

**Keputusan:** `MISSING_DROP_THRESHOLD = 0.60`. Enam kolom dibuang: `total_points_and_fees`,
`discount_points`, `lender_credits`, `prepayment_penalty_term`, `intro_rate_period`,
`multifamily_affordable_units`.

**Alasan:** Ini *missingness struktural*, bukan kualitas data yang buruk. Mengimputasi kolom yang
mayoritas isinya kosong berarti mengarang lebih dari separuh nilainya; distribusi buatan itu akan
mendominasi data asli dan pola apa pun yang "ditemukan" di kolom tersebut sebenarnya adalah pola
imputasi kita sendiri.

**Mengapa 60% dan bukan 20% atau 30%?** Ambang sengaja dibuat longgar. Banyak field HMDA kosong
karena aturan pelaporan (institusi tertentu dikecualikan), bukan karena kesalahan pencatatan.
Ambang ketat 20-30% akan membuang kolom yang masih memuat 70-80% informasi nyata. Selama masih ada
40% nilai asli, kolom dipertahankan.

### 1.2 Mengapa imputasi median, bukan mean?

**Keputusan:** Fitur `CONTINUOUS` yang masih kosong diisi median kolomnya.

**Alasan:** Distribusi `income`, `loan_amount`, dan `property_value` sangat *skewed* ke kanan dengan
outlier ekstrem — dataset ini memuat property value hingga $130 juta. Mean tertarik oleh nilai
raksasa tersebut, sehingga nilai "pusat" yang diisikan menjadi jauh lebih tinggi daripada aplikasi
tipikal. Median tahan terhadap ekor ekstrem sehingga tetap mewakili pemohon kebanyakan.

**Mengapa bukan KNN imputation atau model imputasi?** Tiga alasan:

1. **Biaya.** Mencari tetangga terdekat pada 100.000 baris jauh lebih mahal daripada satu statistik ringkas.
2. **Auditabilitas.** Untuk data regulasi, "diisi dengan median kolom" dapat dipertanggungjawabkan
   ke regulator. Nilai hasil model sulit dijelaskan dan sulit direproduksi.
3. **Risiko sirkularitas.** Imputasi berbasis fitur lain menanamkan korelasi buatan antar fitur.
   Korelasi itu kemudian bisa "ditemukan kembali" sebagai association rule di Fase 3 — kita akan
   menemukan pola yang kita ciptakan sendiri.

### 1.3 Mengapa kategorikal diisi "Unknown", bukan modus?

**Alasan:** Kekosongan pada field kategorikal HMDA bermakna — sering menandakan `Exempt` atau memang
tidak dilaporkan. Mengisi dengan modus menghapus sinyal itu sekaligus menggelembungkan kategori
terbanyak secara artifisial. `"Unknown"` mempertahankannya sebagai kategori tersendiri yang tetap
bisa dianalisis.

### 1.4 Mengapa menambahkan indikator `_was_missing`?

**Alasan:** Agar jejak imputasi terekam. Tanpa indikator, nilai hasil imputasi tidak bisa dibedakan
dari nilai asli. Indikator ini sengaja **dikeluarkan** dari input Fase 2-4 karena sifatnya diagnostik
proses, bukan karakteristik pemohon. Hasil akhir: 0 sel kosong tersisa setelah pembersihan.

### 1.5 Mengapa fitur dipartisi menjadi lima tipe?

| Tipe | Isi | Perlakuan |
|---|---|---|
| `CONTINUOUS` | income, loan_amount, property_value, CLTV | imputasi median → penskalaan |
| `STRING_BAND` | debt_to_income_ratio, applicant_age | dipertahankan sebagai band |
| `CATEG_CODE` | action_taken, loan_purpose, occupancy_type | kode → label bermakna |
| `TEXT_CATEG` | state_code, derived_race | dipakai apa adanya |
| `IDS` | lei, census_tract | dikecualikan dari pemodelan |

**Alasan:** Tiap tipe menuntut perlakuan berbeda, dan salah perlakuan menciptakan kesalahan diam-diam.
Memperlakukan `CATEG_CODE` (misalnya `loan_purpose` bernilai 1/2/3) sebagai angka kontinu menciptakan
*ordinality palsu* — seolah tujuan 3 bernilai "tiga kali" tujuan 1. Partisi ini divalidasi harus
menutup **setiap** kolom, sehingga tidak ada fitur yang terlewat atau diperlakukan ganda.

### 1.6 Mengapa fitur pasca-keputusan dikecualikan?

**Alasan:** Mencegah kebocoran data (*leakage*). Field seperti `interest_rate`, `total_loan_costs`,
dan `origination_charges` baru ada **setelah** aplikasi disetujui. Memakainya untuk menjelaskan
persetujuan berarti menebak hasil dari hasil: akurasinya tampak tinggi tetapi tidak berguna, karena
saat aplikasi baru masuk field tersebut belum ada.

### 1.6b Audit multikolinearitas: VIF dan uji rekonstruksi DTI

Cek korelasi berpasangan hanya melihat dua fitur sekaligus, sehingga buta terhadap
redundansi yang baru muncul ketika beberapa fitur digabung. Sebuah fitur bisa berkorelasi
rendah terhadap setiap fitur lain satu per satu, tetapi tetap dapat diprediksi hampir
sempurna dari kombinasi beberapa fitur lainnya.

**VIF (Variance Inflation Factor).** Dihitung pada 17 fitur numerik dari 67.827 baris
lengkap. Karena VIF = 1 / (1 − R²), ambang VIF > 10 persis setara dengan R² > 0,90, yaitu
ambang yang sama dengan cek berpasangan namun diterapkan pada dimensi yang benar.

| Fitur | VIF |
|---|---|
| tract_owner_occupied_units | 6,00 |
| tract_one_to_four_family_homes | 4,05 |
| tract_population | 3,58 |
| any_exempt_field | 2,95 |
| property_value_was_missing | 2,61 |

**Hasil: 0 fitur melewati ambang**, konsisten dengan 0 pasangan pada cek berpasangan.

**Uji terarah: dapatkah DTI direkonstruksi dari fitur ukuran?** DTI sudah diubah menjadi
band sehingga tidak masuk ruang fitur numerik dan tidak muncul di tabel VIF, padahal
justru DTI yang paling dicurigai tumpang tindih dengan income dan besar loan. Band DTI
dipetakan ke titik tengahnya, lalu direkonstruksi dari income, besar loan, nilai properti,
dan rasio loan terhadap income. Rasio dibentuk eksplisit karena bentuk tumpang tindih yang
dipertanyakan memang rasio, dan korelasi Pearson pada kolom mentah tidak dapat melihatnya.

| Prediktor | Spearman terhadap DTI |
|---|---|
| loan_to_income | +0,373 |
| log_income | −0,330 |
| log_loan_amount | +0,057 |
| log_property_value | +0,007 |

**Hasil: R² = 0,100, setara VIF 1,11.** DTI **tidak** dapat direkonstruksi dari fitur
ukuran. Sisa variasi 90% sejalan dengan tiga alasan struktural: utang non-hipotek tidak
terekam di HMDA, besar cicilan bergantung pada bunga dan tenor, dan DTI hanya diterbitkan
sebagai tujuh tingkat ordinal. Keduanya membawa informasi berbeda sehingga sama-sama
dipertahankan.

### 1.7 Mengapa winsorize 1%/99%, dan mengapa hanya pada salinan clustering?

**Alasan:** K-Means meminimalkan jarak kuadrat, sehingga satu properti $130 juta dapat menarik
seluruh centroid dan membuat cluster tak bermakna. Winsorize mengekang ekor **tanpa membuang baris**.

Ini hanya diterapkan pada salinan untuk clustering. Nilai asli tetap utuh untuk profiling,
association rules, dan deteksi anomali — Fase 4 justru **membutuhkan** nilai ekstrem itu.

### 1.8 Mengapa StandardScaler untuk clustering tetapi RobustScaler untuk anomali?

Ini keputusan yang paling sering disalahpahami, karena tujuan keduanya **berlawanan**.

| | Clustering (Fase 2) | Deteksi anomali (Fase 4) |
|---|---|---|
| Scaler | `StandardScaler` (z-score) | `RobustScaler` (median & IQR) |
| Statistik | mean, standar deviasi | median, IQR |
| Tujuan | setiap fitur berkontribusi setara | outlier harus tetap menonjol |

**Mengapa StandardScaler untuk clustering:** jarak Euclidean menuntut skala setara. Tanpa penskalaan,
`loan_amount` (ratusan ribu) akan menenggelamkan `CLTV` (persen) — jaraknya praktis hanya mengukur
loan_amount.

**Mengapa RobustScaler untuk anomali:** mean dan standar deviasi **tertarik oleh outlier**. Bila
dipakai, outlier ekstrem menggelembungkan std sehingga z-score-nya sendiri mengecil — outlier
"menyembunyikan diri". Median dan IQR tidak sensitif terhadap nilai ekstrem, sehingga anomali sejati
tetap jauh dari pusat.

---

## Fase 2 — Segmentasi (Clustering)

### 2.1 Perbandingan empat metode

Semua metrik diukur pada matriks berskala yang benar-benar dilihat algoritmanya.

| Metode | Cakupan | Cluster | Noise | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI vs K-Means |
|---|---|---|---|---|---|---|---|
| **K-Means** | Seluruh data (99.995) | 7 | 0 | **0,303** | 1,146 | 21.867 | 1,000 |
| DBSCAN | Sampel 20.000 | 17 | 895 | 0,297 | 1,090 | 1.810 | 0,849 |
| Hierarchical (Ward) | Sampel 4.000 | 7 | 0 | 0,294 | 1,152 | 837 | 0,906 |
| CLARANS (k-medoids) | Sampel 4.000 | 7 | 0 | 0,245 | 1,242 | 768 | 0,710 |
| *K-Means (pada sampel 4.000)* | Sampel 4.000 | 7 | 0 | **0,303** | **1,137** | **869** | 1,000 |

**Cara membaca:** Silhouette makin tinggi makin baik (cluster rapat dan terpisah). Davies-Bouldin
makin rendah makin baik. Calinski-Harabasz makin tinggi makin baik. ARI mengukur kesepakatan dengan
K-Means (1,0 = identik, 0 = acak).

### 2.2 Mana yang terbaik, dan mengapa?

**K-Means.** Pada sampel 4.000 baris yang sama — pembanding setara — K-Means unggul di **ketiga**
metrik sekaligus: silhouette tertinggi (0,303 vs Ward 0,294 vs CLARANS 0,245), Davies-Bouldin
terendah (1,137), dan Calinski-Harabasz tertinggi (869). Keunggulannya bukan hanya pada satu metrik
yang kebetulan menguntungkan.

Ditambah satu alasan praktis yang menentukan: **hanya K-Means yang bisa dijalankan pada seluruh
99.995 aplikasi.**

**Catatan kejujuran metrik:** baris DBSCAN diukur pada sampel 20.000 dan mengecualikan noise, jadi
angkanya **tidak sebanding langsung** dengan baris sampel 4.000. Davies-Bouldin DBSCAN yang terlihat
paling rendah (1,090) sebagian merupakan konsekuensi dari mengeluarkan 895 titik tersulit dari
perhitungan. Karena itu baris "K-Means (pada sampel 4.000)" disertakan sebagai pembanding setara.

### 2.3 Mengapa K = 7, atas dasar apa?

**Keputusan:** `best_k = sil_k` — K dipilih dari **silhouette tertinggi**, bukan dari elbow.

**Alasan:** Elbow hanya memberi kandidat dan penafsirannya subjektif ("di mana sikunya?"). Dua orang
bisa membaca kurva inertia yang sama dan memilih K berbeda. Silhouette mengukur langsung seberapa
rapat sebuah titik terhadap cluster-nya dibandingkan cluster terdekat, sehingga menghasilkan satu
angka yang bisa dibandingkan antar K secara objektif. Elbow tetap dihitung dan dilaporkan sebagai
pemeriksaan silang, tetapi keputusan akhir ada pada silhouette.

### 2.4 Mengapa tetap menjalankan tiga metode lain?

| Metode | Alasan dijalankan |
|---|---|
| **DBSCAN** | K-Means memaksa setiap titik masuk cluster dan mengasumsikan bentuk membulat. DBSCAN berbasis densitas sehingga menemukan bentuk sembarang, dan secara eksplisit menandai *noise*. 895 titik noise itu menjadi salah satu dari lima detektor pada ensemble anomali Fase 4. |
| **Hierarchical (Ward)** | Tidak perlu menetapkan K di awal; membangun hierarki penuh yang bisa dipotong pada K berapa pun. Pertanyaannya berubah menjadi "apakah struktur 7 segmen memang ada di data?" — ARI 0,906 menjawab ya. |
| **CLARANS (k-medoids)** | Pusat cluster berupa **aplikasi nyata** (medoid), bukan rata-rata sintetis. Centroid K-Means bisa berupa titik yang tak pernah ada; medoid bisa ditunjuk sebagai contoh konkret. Medoid juga lebih robust terhadap outlier. |

**Interpretasi ARI:** Ward menyepakati K-Means dengan kuat (0,906), menegaskan segmentasi bukan
artefak satu algoritma. CLARANS lebih rendah (0,710) — dan itu **wajar**, bukan kegagalan: CLARANS
mengoptimalkan medoid, bukan centroid, sehingga tujuan optimasinya memang berbeda.

### 2.5 Mengapa DBSCAN dan hierarchical disampel, bukan seluruh data?

Ini keputusan teknis, **bukan** karena algoritmanya menyampel sendiri.

- **Hierarchical wajib disampel.** Ward membangun matriks linkage berpasangan: kompleksitas memori
  O(n²). Pada 100.000 baris itu sekitar 10 miliar pasangan — puluhan GB RAM, dipastikan kehabisan memori.
- **DBSCAN bisa lebih besar,** tetapi menjadi lambat dan penyetelan `eps` makin sensitif pada skala
  besar. Sampel 20.000 sudah cukup untuk mengukur struktur densitas.
- **CLARANS** disampel untuk kecepatan pencarian medoid.

---

## Fase 3 — Association Rules

### 3.1 Ringkasan ambang

| Parameter | Nilai | Hasil |
|---|---|---|
| `min_support` | 2% | ≈1.357 aplikasi |
| `max_len` | 3 item | membatasi ledakan kombinatorial |
| `min_lift` | 1,2 | menyaring asosiasi sepele |
| `min_confidence` | 55% | mengalahkan base rate |
| `min_improvement` | 2 poin | memangkas varian trivial |

**Alur penyaringan:** 28 aturan kandidat (21 Denied + 7 Originated) → **11 aturan final**
(8 Denied + 3 Originated).

### 3.2 Mengapa minimum support 2%?

Pada 67.827 aplikasi berkeputusan, 2% berarti sekitar **1.357 aplikasi**. Angka ini cukup besar agar
estimasi confidence stabil dan tidak lahir dari segelintir kasus, tetapi cukup kecil agar segmen
minoritas yang penting — misalnya *manufactured housing* — tidak lenyap sebelum sempat dianalisis.
Ambang ini sekaligus menahan ledakan kombinatorial jumlah itemset.

### 3.3 Mengapa lift harus > 1,2?

Lift = 1,0 berarti antecedent dan konsekuen **statistik independen**: aturannya tidak memberi
informasi apa pun. Ambang 1,2 menuntut kejadian bersama minimal **20% lebih sering daripada
kebetulan**, sehingga menyaring asosiasi sepele yang muncul hanya karena kedua item sama-sama umum.
Tanpa ambang ini, aturan seperti "First_Lien → Originated" akan lolos semata karena mayoritas
aplikasi memang first lien dan mayoritas memang disetujui.

### 3.4 Mengapa confidence minimal 55%?

Karena aturan harus **mengalahkan tebakan dasar**. Base rate-nya 23,1% Denied dan 76,9% Originated.

Untuk aturan penolakan, 55% menempatkannya jauh di atas base rate 23,1% — lebih dari dua kali lipat.
Ambang ini juga melewati batas mayoritas sederhana: bila aturan menyatakan "Denied", lebih dari
separuh kasus yang cocok memang benar-benar ditolak, sehingga aturannya layak ditindaklanjuti.

### 3.5 Mengapa panjang itemset dibatasi maksimal 3?

Dua alasan: **interpretabilitas** dan **biaya**. Aturan dengan 5-6 kondisi praktis tidak bisa
dioperasionalkan oleh tim underwriting, dan jumlah kombinasi tumbuh eksplosif seiring panjang
itemset. Tiga kondisi cukup untuk menangkap interaksi bermakna (misalnya DTI + jenis lien + jenis
loan) tanpa menjadi tak terbaca.

### 3.6 Mengapa ada improvement filter, dan mengapa 28 menjadi 11?

**Masalahnya:** lolos ambang saja tidak berarti sebuah aturan menambah pengetahuan. Menambahkan
`derived_race=White` atau `lien_status=First_Lien` ke aturan `DTI>60% → Denied` menghasilkan aturan
"baru" yang tetap lolos semua ambang, padahal confidence-nya nyaris sama dengan aturan induknya.
Itu hanya pengulangan dengan kata tambahan — dan berbahaya, karena bisa dibaca seolah ras merupakan
faktor penolakan padahal ia tidak menambah daya pisah apa pun.

**Solusinya:** improvement filter bergaya Bayardo. Sebuah aturan hanya dipertahankan bila
confidence-nya mengungguli **setiap** proper sub-rule dengan konsekuen sama, minimal 2 poin
persentase. Hasilnya 28 kandidat menyusut menjadi 11 aturan yang benar-benar berbeda satu sama lain.

### 3.7 Mengapa aturan diuji signifikansinya?

Support dan confidence tidak memberi tahu apakah pola bisa muncul secara kebetulan. Karena itu
ditambahkan **chi-square** (menguji apakah asosiasinya berbeda nyata dari independen) dan **Wilson
confidence interval** (memberi rentang ketidakpastian confidence yang tetap layak pada n kecil
maupun proporsi mendekati 0 atau 1, tidak seperti interval normal biasa).

---

## Fase 4 — Deteksi Anomali

### 4.1 Ringkasan lima detektor

| Detektor | Ambang | Filosofi |
|---|---|---|
| IQR | ≥3 fitur di luar 1,5×IQR | Global |
| Z-score | ≥1 fitur dengan \|z\| > 3 | Global |
| Isolation Forest | contamination 1%, 200 pohon | Global |
| Local Outlier Factor | 20 tetangga, contamination 1% | Kontekstual |
| DBSCAN-noise | titik noise dari Fase 2 | Kontekstual |

**Hasil:** 739 rekaman mendapat ≥3 suara. Taksonomi: 88.971 Normal (88,98%), 9.959 outlier global
(9,96%), 589 kontekstual/lokal (0,59%), 476 keduanya (0,48%).

### 4.2 Mengapa lima detektor sekaligus?

Karena tiap metode punya titik buta, dan **dua filosofi deteksi menangkap hal yang berbeda**:

- **Global** (IQR, Z-score, Isolation Forest) mencari nilai yang menyimpang dari distribusi seluruh
  dataset, tanpa syarat — misalnya property value $130 juta, ekstrem apa pun konteksnya.
- **Kontekstual** (LOF, DBSCAN-noise) mencari rekaman yang tampak normal pada setiap ambang fitur
  tunggal, tetapi aneh dibandingkan tetangganya.

Angka taksonomi membuktikan keduanya memang berbeda: hanya 476 rekaman ditandai kedua filosofi,
sementara 9.959 hanya global dan 589 hanya kontekstual. Memakai satu pendekatan saja akan melewatkan
seluruh kelas anomali yang lain.

### 4.3 Mengapa ambang konsensus 3 dari 5?

Untuk menekan *false positive*. Metode tunggal mudah menandai rekaman yang sebenarnya wajar, tetapi
kesepakatan mayoritas dari lima detektor yang bekerja dengan prinsip berbeda jauh lebih sulit terjadi
secara kebetulan. Ambang ini juga memaksa setidaknya satu metode global dan satu kontekstual ikut
setuju pada sebagian besar kasus.

### 4.4 Mengapa contamination 1%?

Ini **asumsi operasional** tentang porsi rekaman yang layak ditinjau manual, bukan klaim bahwa tepat
1% data itu salah. Nilainya dipilih agar antrean tinjauan tetap realistis bagi tim manusia. Parameter
ini menentukan ambang skor, bukan kebenaran suatu rekaman — itulah sebabnya keputusan akhir tetap
melalui triase.

### 4.5 Mengapa 15 rekaman teratas ditriase manual, bukan langsung dibuang?

**Karena nilai ekstrem tidak sama dengan salah.** Jumbo loan $9 juta itu ekstrem tetapi sah;
CLTV 900% mustahil secara aritmetika. Membedakannya butuh uji konsistensi internal: apakah
CLTV × property value konsisten dengan loan ini ditambah senior lien, dan apakah income yang
dilaporkan masuk akal untuk membayar utangnya.

Hasil triase menunjukkan mengapa ini penting: seluruh 15 rekaman teratas berakhir dengan verdict
**RARE BUT VALID** — tidak satu pun kesalahan data. Bila ensemble dipakai sebagai aturan penghapusan
otomatis, 15 aplikasi sah akan terbuang.

### 4.6 Mengapa outlier kolektif dilaporkan terpisah?

Kelima detektor menilai rekaman **satu per satu**, sehingga secara desain buta terhadap pola
kelompok. Fakta bahwa 100% dari 2.413 loan ≥ $1 juta berakhiran `"...5.000"` tidak membuat satu pun
loan itu janggal; yang janggal adalah **polanya secara kolektif**. Itu tanda aturan pembulatan
HMDA — karakteristik proses pembangkitan data, bukan sinyal risiko.

---

## Fase 5 — Pelaporan dan Keadilan

### 5.1 Mengapa temuan keadilan disebut asosiasi, bukan sebab-akibat?

Selisih persetujuan antar tract minoritas tidak menutup setelah stratifikasi DTI — terbesar 12,1
poin persentase justru pada kelompok berisiko terendah. Ini **sinyal untuk diselidiki**, bukan bukti
diskriminasi, karena data publik HMDA **tidak memuat** variabel underwriting penting: credit score,
cadangan dana, riwayat pembayaran, dan konteks kebijakan lender. Tanpa variabel itu, faktor perancu
tidak bisa dikesampingkan.

### 5.2 Mengapa What-If berupa pencarian historis, bukan model prediktif?

What-If menampilkan tingkat persetujuan historis nyata di antara aplikasi yang cocok dengan seluruh
atribut terpilih. Ini disengaja: hasilnya dapat ditelusuri ke aplikasi nyata dan tidak menyiratkan
jaminan bagi pemohon individual. Model prediktif akan memberi kesan presisi yang tidak didukung data
publik ini (lihat 5.1) dan berisiko dipakai seolah keputusan underwriting.

---

## Lampiran — Ringkasan seluruh keputusan

| # | Keputusan | Pilihan | Alasan inti |
|---|---|---|---|
| 1 | Ambang buang kolom | >60% kosong | Missingness struktural; longgar agar field pelaporan tetap terpakai |
| 2 | Imputasi kontinu | Median | Tahan skew & outlier; auditable |
| 3 | Imputasi kategorikal | "Unknown" | Kekosongan bermakna (Exempt) |
| 4 | Jejak imputasi | `_was_missing` | Auditabilitas; dikecualikan dari pemodelan |
| 5 | Partisi fitur | 5 tipe | Cegah ordinality palsu; validasi cakupan penuh |
| 6 | Fitur pasca-keputusan | Dikecualikan | Cegah leakage |
| 6b | Audit multikolinearitas | VIF ambang 10 | Cek berpasangan buta terhadap redundansi multivariat; 0 fitur melewati ambang |
| 6c | Uji rekonstruksi DTI | Dipertahankan | R² hanya 0,100, DTI membawa informasi yang tidak ada di fitur ukuran |
| 7 | Winsorize 1/99% | Hanya salinan clustering | Cegah centroid tertarik outlier; nilai asli utuh untuk Fase 4 |
| 8 | Scaler clustering | StandardScaler | Jarak Euclidean butuh fitur setara |
| 9 | Scaler anomali | RobustScaler | Median/IQR tak terdistorsi outlier yang dicari |
| 10 | Metode utama | K-Means | Unggul 3 metrik + satu-satunya yang skalabel penuh |
| 11 | Jumlah cluster | K=7 | Silhouette tertinggi (objektif), elbow sebagai cek silang |
| 12 | Sampling DBSCAN/Ward | 20k / 4k | Ward O(n²) memori wajib disampel |
| 13 | Min support | 2% (~1.357) | Stabil tetapi tidak menghapus segmen minoritas |
| 14 | Min lift | 1,2 | 20% di atas kebetulan; buang asosiasi sepele |
| 15 | Min confidence | 55% | Di atas base rate 23,1% dan di atas mayoritas |
| 16 | Panjang itemset | ≤3 | Interpretabilitas & biaya |
| 17 | Improvement filter | ≥2 poin | Buang varian trivial; 28 → 11 |
| 18 | Uji signifikansi | Chi-square + Wilson CI | Pastikan pola bukan kebetulan |
| 19 | Jumlah detektor | 5 (global + kontekstual) | Tiap metode punya titik buta |
| 20 | Ambang konsensus | ≥3 suara | Tekan false positive |
| 21 | Contamination | 1% | Asumsi operasional antrean tinjauan |
| 22 | Triase | 15 teratas manual | Ekstrem ≠ salah; butuh uji konsistensi |
| 23 | Klaim keadilan | Asosiasi saja | Data publik tanpa credit score/cadangan |

---

*Seluruh angka pada dokumen ini dihasilkan oleh `python app/build_data.py` dan tersimpan di
`data/processed/`. Tabel perbandingan clustering berasal dari `dash_clustering_comparison.csv`.*
