import streamlit as st

import pandas as pd
import numpy as np
import pickle
import json
import time


#load the files!
with open('model_Cat.pkl', 'rb') as file_1 : 
    model_Cat = pickle.load(file_1)

with open('cat_terbaik.pkl', 'rb') as file_2 : 
    cat_terbaik = pickle.load(file_2)
    
    

def run():
    # judul
    st.title('Prediksi Deposit jangka panjang')
    
    with st.form('Prediction Term Deposit '):
        st.title('Term Deposit')
        st.write('#### Silahkan Masukan Data Diri Anda**')
        full_name = st.text_input('Full Name', help='Masukan Nama Lengkap anda', value='')
        age=st.number_input('age',min_value=0,max_value=100,step=1,help='Usia Nasabah?')
        job=st.selectbox('job',['admin','technician','services','management',
                                'retired','blue-collar','unemployed','entrepreneur',
                                'housemaid','unknown','self-employed','student'],index=1,help='Masukan Pekerjaan yang tertera.')
        marital=st.selectbox('marital',['married','single','divorced'],index=1,help='Masukan Status Pernikahan')
        education=st.selectbox('education',['secondary','tertiary','primary','unknown'],index=1,help='Masukan Pendidikan anda')
        st.markdown('--------------')
        st.write('#### Silahkan Masukkan Informasi Anda')
        default=st.selectbox('default',['no','yes'],index=1)
        balance=st.number_input('balance', min_value=0,max_value=100000,step=1,help='Nominal Nasabah?')
        housing=st.selectbox('housing',['no','yes'],index=1)
        loan=st.selectbox('loan',['no','yes'],index=1)
        contact=st.selectbox('contact',['cellular','telephone','unknown'],index=1)
        day=st.number_input('day',min_value=0,max_value=31,step=1,help='Waktu terakhir kali dihubungin?')
        month=st.selectbox('month',['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],index=1)
        
        st.markdown('----------------')
        duration=st.slider('duration',min_value=0,max_value=10000,value=0,help='Berapa lama komunikasi dengan calon nasabah?, durasi dalamm detik')
        campaign=st.slider('campaign',min_value=0,max_value=60,value=0,help='Jumlah melakukan kontak pada calon nasabah')
        pdays=st.slider('pdays',min_value=0,max_value=854,value=0,help='Jumlah hari terakhir kali menghubungin calon nasabah')
        previous=st.slider('previous',min_value=0,max_value=54,value=0,help='Kontak yang sudah dilakukan')
        poutcome=st.selectbox('poutcome',['success','failure','other','unknown'],index=1,help='Apakah hasilnya ?')
        
        st.markdown('-------')
        submit = st.form_submit_button('Predict')
        
    if submit:
     data_inf = {
         
        'name': full_name,
        'age' : age,
        'job' : job,
        'marital' : marital,
        'education' : education,
        'default' : default,
        'balance' : balance,
        'housing' : housing,
        'loan' : loan,
        'contact' : contact,
        'day': day,
        'month' : month,
        'duration' : duration,
        'campaign' : campaign,
        'pdays' : pdays,
        'previous' : previous,
        'poutcome': poutcome,
        }
    
     data_inf = pd.DataFrame([data_inf])
      
     prediksi = cat_terbaik.predict(data_inf)
     
    with st.spinner('Apakah calon nasabah akan melakukan deposit?'):
     time.sleep(7)
    st.success('Prediksi Selesai!')
        
    if submit:
     prediksi = cat_terbaik.predict(data_inf) 
     if prediksi == 'no':
        st.write('TIDAK, adalah jawabannya')
     elif prediksi == 'yes':
        st.write('Selamat, jawabannya adalah IYA')
         
if __name__ == '__main__':
    run()
     
    