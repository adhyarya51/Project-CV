import streamlit as st

import eda
import prediction


navigation = st.sidebar.selectbox("Select Page", 
                                  options=['Exploratory Data Analysis', 'Prediction Deposit'])
st.sidebar.write('# About')
st.sidebar.write('''
Deposito jangka panjang adalah sebuah salah satu produk perbankan dimana nasabah menyetorkan sejumlah uang
untuk disimpan alam kurun waktu yang lama di bank. Pada halaman ini kami memberikan informasi gambaran besar
bentuk suatu data, dan sistem proses prediksi deposito. 

Halaman tersebut dibagi menjadi dua yaitu : 
- Exploratory Data Analysis
- Prediction Term Deposito
                 ''')


if navigation == 'Exploratory Data Analysis':
    eda.run()
else:
    prediction.run()