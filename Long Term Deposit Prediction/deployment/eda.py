import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
import plotly.express as px
from PIL import Image




st.set_page_config(
    page_title= 'Data Bank Portugies',
    layout= 'wide',
    initial_sidebar_state= 'expanded'
)


def run():
    # membuat title 
    st.title ('Bank : Prediksi Deposit jangka panjang')
    
    # buat sub header 
    st.subheader('Exploratory Data Analysis')
    
    # membuat gambar 
    st.image('https://media.istockphoto.com/id/903334106/id/foto/cabang-coimbra-dari-bank-sentral-portugis-banco-de-portugal.jpg?s=612x612&w=0&k=20&c=PFS3ilN6DrS-SyJ8_99dczqZE4UeGm_AS4ofklfo4WQ=')
        
    # deskripsi 
    st.write('# Latar Belakang ')
    st.write('''
    
    Data yang berasal dari salah satu institusi bank asal portugis. 
    Sebuah kampanye yang berdasarkan dari panggilan telepon. 
    Tim pemasaran ingin melakukan prediksi terhadap calon nasabah akan melakukan langganan 
    atau tidak melalui deposit. Hal itu dilakukan dengan melakukan analisa pola yang akan 
    membantu tim dalam menentukan strategi.''')
    
    st.write('# Problem Statement')
    
    st.write('''
    Penentuan prediksi model ditentukan dengan
    analisa pola untuk mengetahui pola klien.''')
    
    st.divider()
    # mencoba menampilkan dataframe
    st.write('Data Bank Portugies')
    df = pd.read_csv('bank.csv')
    st.dataframe(df.sample(50))
    
    # Visualisasi
    st.write('## Exploratory Data Analysis')
    # menampilkan chart
    ## Barplot
    st.write('### Grafik berdasarkan Pekerjaan, Pernikahan, dan Umur')
    # melakukan grouping 'job' dan 'marital', menghitung rata2 'age'
    df_grouped = df.groupby(['job', 'marital'])['age'].mean().reset_index()

    # membuat bar chart
    fig = px.bar(df_grouped, x='job', y='age', color='marital',
                title='rata - rata umur terhadap Pekerjaan, dan Pernikahan',
                labels={'job': 'Pekerjaan', 'age': 'rata - rata umur'})
    # menentukan layout
    fig.update_layout(xaxis_tickangle=30, xaxis_title='Job', yaxis_title='Average Age')
    fig.update_traces(marker_line_color='blue', marker_line_width=1.5)  # Adding borders to bars
    # menampilkan bar chart
    st.plotly_chart(fig)
    
    st.write('''
    Hasil observasi data : 
    - pada sebaran data, bahwa mayoritas pensiun lebih banyak, dibagi menjadi tiga, `menikah`, `single`, dan `cerai`. 
    Diketahui rata - rata umurnya adalah **66**,**65**, dan **58**. 
    - sedangkan, data terendah pada murid, diketahui rata - rata umurnya dari `menikah`, `single`, dan `cerai` 
    yaitu : **37**,**31**, dan **25**''')
    
    # memanggil dataframe
    df = pd.DataFrame(df)

    st.write('### Table klasifikasi')
 
    # membuat kelas
    Tingkatan = []
    # membentuk looping
    for i in df['age']:
        if i >= 5 and i < 11:
            Tingkatan.append('Anak - anak')
        elif i >= 12 and i < 25:
            Tingkatan.append('Remaja')
        elif i >= 26 and i < 45:
            Tingkatan.append('Dewasa')
        else:
            Tingkatan.append('Lansia')

    df['Klasifikasi'] = Tingkatan
    st.write(df)
    

    st.write('### Grafik Klasifikasi')
    
    # membuat bar chart
    calculate = df.groupby('Klasifikasi')['Klasifikasi'].count().sort_values(ascending=False)

    # melakukan plotting
    plt.figure(figsize=(10, 5))
    sns.barplot(x=calculate.index, y=calculate.values)
    plt.title('Tingkatan Umur Chart')
    plt.xlabel('Umur')
    plt.ylabel('Count')
    plt.show()
    st.pyplot(plt.gcf())
    
    
    st.write('''Observasi : 
    - Keseluruhan data dilakukan klasifikasi dan ditemukan bahwa ada tiga kelas umur yaitu : `Dewasa`, `Lansia`, `Remaja`    
    - Mayoritas bar chart menunjukan `Dewasa` sebagai nasabah terbanyak, `Lansia`, dan `Remaja`.    
    ''')
    
    #---
    
    st.write('### Grafik Umur, Pekerjaan terhadap Saldo')
    # set ukuran figure 
    # Create Bar chart

    calculate_klasifikasi = df.groupby(['Klasifikasi'])['balance'].mean().reset_index()
    calculate_pekerjaan = df.groupby(['job'])['balance'].mean().reset_index()
    # plot utama
    plt.figure(figsize=(10, 5))
    # membuat sub plot
    plt.subplot(1,2,1)
    sns.barplot(data=calculate_klasifikasi, x='Klasifikasi', y='balance')
    plt.title('Klasifikasi Usia terhadap balance')
    plt.xlabel('Klasifikasi')
    plt.ylabel('balance average')
    plt.legend(title='balance')
    plt.xticks(rotation=90)

    # membuat sub plot
    plt.subplot(1,2,2)
    sns.barplot(data=calculate_pekerjaan, x='job', y='balance')
    plt.title('Klasifikasi Pekerjaan terhadap balance')
    plt.xlabel('Pekerjaan')
    plt.ylabel('balance average')
    plt.legend(title='balance')
    plt.xticks(rotation=90)

    # menampilkan chart
    plt.show()
    st.pyplot(plt.gcf())
    
    st.write('''
    
    Hasil : 

    - dalam klasifikasi usia diketahui Lansia memiliki rata - rata saldo tertinggi, dibandingkan Dewasa, dan Remaja
    - untuk pekerjaan ditemukan nasabah pensiunan memiliki saldo tertinggi rata - rata sebesar 2000 - 2500, 
    serta rata - rata saldo terendah adalah pekerjaan services. 
    
    ''')
    
    #-----
    
    st.write('### Grafik Pernikahan')
    
    # membuat Bar chart
    calculate = df.groupby('marital')['marital'].count().sort_values(ascending=False)
    # membentuk grafik
    plt.figure(figsize=(10, 5))
    sns.barplot(x=calculate.index, y=calculate.values)
    plt.title('Grafik Pernikahan')
    plt.xlabel('Pernikahan')
    plt.ylabel('Count')
    # menampilkan plot
    plt.show()
    st.pyplot(plt.gcf())
    
    st.write('''
             Hasil : 
    - jumlah nasabah yang `menikah` lebih dari **6000**, 
    dan `single` antara **3000** - **4000**, dan yang `bercerai` antara **1000**. 
    ''')
    
    st.write('### Grafik Pendidikan')
    
     # membuat Bar chart
    calculate = df.groupby('education')['education'].count().sort_values(ascending=False)
    # membentuk grafik
    plt.figure(figsize=(10, 5))
    sns.barplot(x=calculate.index, y=calculate.values)
    plt.title('Grafik Pendidikan')
    plt.xlabel('Pendidikan')
    plt.ylabel('Count')
    # menampilkan plot
    plt.show()
    st.pyplot(plt.gcf())
    
    st.write('''
             Hasil : 
    - dalam sebaran menunjukan mayoritas data pendidikan berurut dari secondary, 
    tertiary, primary, dan unknown.
    ''')
    
    st.write('### Grafik sebaran Kontak')
    # kalkulasi grup
    calculate = df.groupby('contact')['contact'].count().sort_values(ascending=False)
    # membentuk plot
    plt.figure(figsize=(10, 5))
    sns.barplot(x=calculate.index, y=calculate.values)
    plt.title('Grafik kontak')
    plt.xlabel('kontak')
    plt.ylabel('Count')
    # menampilkan plot
    plt.show()
    st.pyplot(plt.gcf())
    
    st.write('''
    Hasil : 
    - Sebaran Kontak menunjukan mayoritas menggunakan cellular dan telephone, serta beberapa belum diketahui. 
             ''')
    
    
    st.write('''### Grafik Balance dan Umur''')
    
    # Membuat scatter plot PURCHASES dengan CREDIT LIMIT
    fig = px.scatter(df, x='balance', y='age', hover_data=['job'], title='Balance dan Umur')
    fig.update_layout(xaxis_title='Saldo', yaxis_title='Umur')
    # menampilkan scatter
    st.plotly_chart(fig)
    
    st.write('''
    Hasil : 
    - Sebaran menggunakan scatter plot menunjukan mayoritas saldo `10000`, dan umur `74`
    - dengan umur tertinggi diumur `95` `pensiunan` saldonya sebesar `2282`,
    - saldo terbanyak sebesar `81000`, pensiunan dan umur `84`
    - dan ada saldo menyampai **minus** pada umur `49`, pekerjaan `manajement`. 
             ''')
    
    st.write('''### Grafik perbandingan Hutang, dan Cicilan Rumah''')
    # Hitung Cicilan Rumah, dan hutang
    housing_balance = df.groupby('housing')['balance'].max()
    loan_balance = df.groupby('loan')['balance'].max()

    # Plot distribusi cicilan rumah dan hutang dalam satu grafik
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(housing_balance.index, housing_balance.values, color='orange')
    plt.title('cicilan rumah tertinggi')
    plt.xlabel('Housing')
    plt.ylabel('Cicilan Tertinggi')

    # Menambahkan label pada sumbu x
    plt.xticks(rotation=90)  # Rotasi label agar lebih jelas
    plt.tight_layout()  # Menyesuaikan layout agar tidak tumpang tindih


    plt.subplot(1, 2, 2)
    plt.bar(loan_balance.index, loan_balance.values, color='blue')
    plt.title('Hutang tertinggi')
    plt.xlabel('Loan')
    plt.ylabel('Hutang Tertinggi')

    # Menambahkan label pada sumbu x
    plt.xticks(rotation=90)  # Rotasi label agar lebih jelas
    plt.tight_layout()  # Menyesuaikan layout agar tidak tumpang tindih
    st.pyplot(plt.gcf())
    
    st.write('''
    Hasil : 
    - dalam dua grafik menunjukan mayoritas tidak memiliki hutang, dan cicilan rumah
    - mayoritas nasabah memiliki hutang dan cicilan disekitaran 40000 - 50000. 
             ''')
    
    st.write('''### Grafik sebaran Deposit''')
    # kalkulasi total deposit
    calculate = df['deposit'].value_counts()
    # membentuk piechart
    fig = px.pie(values=calculate.values, names=calculate.index, 
                title='Distribusi Deposit')
    # memanggil pie chart
    st.plotly_chart(fig)
    
    st.write('''
    - Distribusi deposit terbagi rata dengan ditanya melakukan deposit sebesar 47%, dan tidak sebenyak 52.6%
             ''')
        
if __name__ == '__main__' :
        run()