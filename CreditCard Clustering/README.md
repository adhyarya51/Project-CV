Dataset : [Credit-Card](./Adhy_Arya.csv)  
Object  : Penggambaran Kartu Kredit konsep clustering data dengan menggunakan Scikit-Learn 

### Problem Statement : 

Pengelompokan(cluster) dengan melihat sebaran data guna memahami tingkat nasabah pemegang kartu kredit.

### Latar Belakang

Data sebaran kartu kredit dengan total data kurang lebih 4475 data memiliki informasi tentang pembelian, limit kredit, pembayaran, dan balance setiap akun. Maka dari itu diperlukannya metode pengelompokan (cluster), tujuan pengelompokan adalah memudahkan saya untuk melihat bagaimana bentuk sebaran nasabah pada kartu kredit tersebut. Saya ingin melihat bagaimana sifat - sifat dari setiap kelompok(cluster) untuk mampu melakukan proses deskripsi pada produk kredit saat ini.

### Overal Analysis

**Balance and Balance Frequency**

**Observasi** : 

visualisasi dilakukan dengan 2D bahwa ditemukan beberapa informasi : 

- total cluster ada empat dari 0 sampai dengan 3 
- untuk cluster 0 nasabah mayoritas memiliki saldo, frekuensi pembelian, serta cicilan yang tinggi, namun tidak melakukan pembahuruan
- cluster 1 dengan saldo, frekuensi pembelian, serta cicilan rendah, tetapi melakukan pembahuruan terbaru dalam saldo
- cluster 2 hampir keseluruhan nasabah memiliki daya beli rendah, dan tidak melakukan pembahuruan terbaru, dan 
- cluster 3 memiliki daya beli rendah, tetapi selalu melakukan pembahuruan saldo nasabah.

**Clustering Purchase and Payments**

Mayoritas keseluruhan nasabah berada pada cluster no. 3 dengan melakukan pembelian paling rendah sebesar 3361 sampai 41000. Cluster 0 jarang melakukan pembelian, dan cluster 1, dan 2 tidak banyak melakukan aktifitas dalam melakukan pembelian. 

**Kesimpulan Observasi**:  
Dapat disimpulkan keunggulan cluster ada pada cluster 0 , dan 3 dimana ada nasabah tidak melakukan pembelian, tetapi pembayaran tinggi, dan nasabah melakukan pembelian, tapi tidak tinggi. Disimpulkan mayoritas nasabah memiliki tinggkat belanja besar.


**Clustering Credit Limit and Installment Purchases**

Hasil clustering menyimpulkan bahwa : 
- clustering 0 memiliki limit dari 0 sampai dengan 10k, limit normal, tetapi jarang melakukan pembelian 
- clustering 1 memiliki limit rendah, tapi masih melakukan pembelian dari 1000 sampai 2000.
- clustering 2 tidak terbentuk, tetapi sebaran tidak tertentu 
- clustering 3 menunjukan sebaran limit normal ke tinggi, serta pembelian termasuk tinggi. 

**Cluster Balance and Purchases**

Kesimpulan clustering ditemukan : 

- cluster 0 mayoritas memiliki saldo tetapi tidak melakukan pembelian 
- cluster 1 sebagian memiliki saldo dan daya beli rendah, dan sebagian memiliki saldo rendah, dan memiliki daya beli rendah
- cluster 2 tidak memiliki saldo sedikit , dan tidak melakukan pembelian 
- cluster 3 mayoritas memiliki saldo serta daya beli tinggi, atau pembelian tinggi.

**Clustering sebaran Transaksi**

Kesimpulan data : 
- Prc full payment dengan cluster 3 lebih banyak dibandingkan cluster lainnnya
- Sedangkan, purchase trx untuk cluster 2 lebih banyak dari yang lainnya

berdasarkan scatter plot 

- cluster 0 mayoritas range purchase trx kurang lebih dari 0 sampai dengan 25, dan persentase variasi dari 0 sampai 0.8
- cluster 1 mayoritas melakukan grouping purchase trx antara 0 - 50, dan persentasi sekitar 0 s/d 1 
- cluster 2 memiliki sebaran mmayoritas lebihh luas dari cluster lainnya purchase trx kurang lebih dari 0 s/d 350, dan persentasi dari 0 s/d 1, 
- cluster 3 membentuk grouping purchase trx antara 0 - 50, dan sebaran persentase dari 0.1 s/d 0.9.


**Clustering Purchases and Credit Limit**

berdasarkan hasil observasi dari scatter plot : 

- Cluster 0 dengan mayoritas limit kredit diatas 3200, dan pembelian dibawah 5000, serta tenure di angka 12
- cluster 1 memiliki ciri - ciri sama dengan cluster 0 
- sedangkan, cluster 3 mayoritas melakuan pembelian 300 s/d 2000, dengan limit kredit sebesar 1000 sampai 15000, dan
- cluster 2 menjadi mayoritas melakukan pembelian dari 1500 sampai diatas 10000,dengan limit kredit bervariasi yaitu di 6000 keatas.

dapat disimpulkan mayoritas nasabah masih dapat melakukan pembelian diatas 1500 dengan limit kredit yang berbeda.

**Total Cluster**

dilakukan penghitungan keseluruhan data dengan mengetahui sebaran cluster keseluruhan. Ditemukan bahwa, cluster 1 lebih unggul, dan diikuti olehh cluster 3, 0 dan 2.



