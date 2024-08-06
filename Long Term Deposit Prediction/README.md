Object : Deployment supervised data dengan memprediksi pola nasabah.

### Problem Statement : 

Penentuan prediksi model ditentukan dengan analisa untuk mengetahui pola klien.

### Latar Belakang : 

Data yang berasal dari salah satu institusi bank asal portugis, ingin membuat sebuah kampanye yang berdasarkan dari panggilan telepon. Saya dan tim pemasaran ingin melakukan prediksi terhadap calon nasabah akan melakukan langganan atau tidak dengan menghubungin calon nasabah melalui telepon. Namun tim ingin mengetahui nasabah yang dihubungin akan berpotensi untuk mendaftar/berlangganan deposito, atau tidak. Maka dari itu pemasaran analisis mencoba prediksi dengan model machine learning dengan analisa pola data yang akan membantu tim dalam menentukan strategi. Berikut informasi detail pada variabel yang telah diketahui.

| Column | Description |
| --- | --- |
| age (numeric) | Amount of given credit in NT dollars (includes individual and family/supplementary credit) |
| job | type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student" "blue-collar","self-employed","retired","technician","services") |
| marital |  marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed) |
| education(categorical) | "unknown","secondary","primary","tertiary" |
| default | has credit in default? (binary: "yes","no") |
| balance | average yearly balance, in euros (numeric) |
| housing | has housing loan? (binary: "yes","no") |
| load | has personal loan? (binary: "yes","no") |
| contact | has personal loan? (binary: "yes","no") |
| day | last contact day of the month (numeric) |
| month | last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec") |
| duration | last contact duration, in seconds (numeric) |
| campaign | number of contacts performed during this campaign and for this client (numeric, includes last contact)|
| pdays| number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)|
| previous | number of contacts performed before this campaign and for this client (numeric)|
| poutcome | outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")|
| target y, pada kasus ini yaitu "deposit" | - has the client subscribed a term deposit? (binary: "yes","no")|

###  Project Ouput 

The project aims to develop a supervised machine learning model using bank data to predict potential customers for long-term deposits. By having 84% accuracy in identifying these customers, the model will help optimize the efforts of frontline callers, reducing the need to contact the same individuals and enhancing overall efficiency repeatedly.
Technology / Tools: Python, Pandas, NumPy, Tableu, Kaggle, SVC, Catboost, grid, pipeline, phik, seaborn, matplotlib.
