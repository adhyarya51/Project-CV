import streamlit as st
import eda
import prediction



navigation = st.sidebar.selectbox("Select Page", 
                                  options=['Exploratory Data Analysis', 'Prediction Klasifikasi Beras'])
st.sidebar.write('# About')
st.sidebar.write('''
Beras adalah salah satu produk makanan pokok paling penting di dunia. Pernyataan ini terutama berlaku di Benua Asia, 
tempat beras menjadi makanan pokok untuk mayoritas penduduk (terutama di kalangan menengah ke bawah masyarakat). 
Benua Asia juga merupakan tempat tinggal dari para petani yang memproduksi sekitar 90% dari total produksi beras dunia. 

Halaman tersebut dibagi menjadi dua yaitu : 
- Exploratory Data Analysis
- Prediction Klasifikasi Beras
                 ''')


if navigation == 'Exploratory Data Analysis':
    eda.run()
else:
    prediction.run()