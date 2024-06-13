import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
import plotly.express as px
import os
import tensorflow as tf 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')



st.set_page_config(
    page_title= 'Data Klasifikasi Beras',
    layout= 'wide',
    initial_sidebar_state= 'expanded'
)

def run():
    # membuat title 
    st.title ('Klasifikasi Data Beras')
    
    # buat sub header 
    st.subheader('Exploratory Data Analysis')
    
    # membuat gambar 
    st.image('data:image/jpeg;base64,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')
        
    # deskripsi 
    st.write('# Latar Belakang ')
    st.write('''
    
    Beras adalah salah satu makanan pokok di dunia, Namun menjadi salah satu paling penting khususnya Benua Asia, dikarenakan tempat beras menjadi makanan pokok untuk mayoritas penduduk(terutama di kalangan menengah kebawah masyarakat). Benua Asia sendiri menjadi tempat tinggal petani, dapat diperkirakan 90% produksi beras oleh petani beras.  

    Namun beras tidak memiliki satu tipe beras, melainkan berbagai macam tipe beras. Klasifikasi beras ternyata memiliki pengaruh pada setiap iklimnya masing - masing negara. Maka dari itu dibutuh model, atau deep learning computer vision untuk mempelajari jenis - jenis beras.

    Pada Kasus ini akan menggunakan data beras yang terdiri dari lima kelas. Kelas masing - masing beras terdiri dari karacadag,ipsala,jasmine,arborio,dan basmati. Harapannya dengan pembentukan model, Hasil deployment dapat melakukan identifikasi beras tentunya menentukan kelima beras tersebut.''')
    
    st.write('# Problem Statement')
    
    st.write('''
    Melakukan klasifikasi beras dan prediksi gambar dengan Computer Vision.''')
    
    st.divider()
       # mencoba menampilkan gambar
    st.write('# Data Beras')

    # fungsi untuk memanggil gambar
    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                img = Image.open(os.path.join(folder, filename))
                if img is not None:
                    images.append((filename, img))
        return images
    # fungsi memplot gambar untuk streamlit
    def plot_images(base_folder_path):
        # Path menuju "Training" subfolder
        training_folder_path = os.path.join(base_folder_path, "training")
        
        # mengambil subfolder dari training
        subfolders = os.listdir(training_folder_path)
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(training_folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue  # lalui apabila bukan folder
            
            st.write(f'Class: {subfolder}')
            
            # Load gambar dari subfolder
            images = load_images_from_folder(subfolder_path)
            
            # memastikan kita hanya mengambil 5 gambar
            images_to_display = images[:5]
            
            fig = plt.figure(figsize=(20, 4))  # penyesuaian gambar
            columns = 5
            rows = 1
            
            for index in range(1, columns * rows + 1):
                if index > len(images_to_display):
                    break
                filename, image = images_to_display[index - 1]
                fig.add_subplot(rows, columns, index)
                plt.imshow(image)
                plt.axis("off")
                plt.title(filename, fontsize=8)
            
            st.pyplot(fig)
            plt.close(fig)  # menutupkan gambar untuk memori

    # judul
    st.write("### menunjukan kelas beras berbeda")

    # lebih spesifik untuk folder
    folder_path = r"Rice_Image_Dataset"  # Main folder path memiliki "training"

    # Plot images dari setiap subfolder didalam "training"
    plot_images(folder_path)
    
    st.write(''' 
                **Informasi bentuk Gambar**
    - ketiga gambar yang dipisahkan sementar dan dibentuk, masing2 memiliki bentuk yang sama yaitu panjang , dan lebar sebesar **224**. Hal ini akan menjadi acuan sebagai parameter untuk modeling.

    **Informasi Dataset,Sampel,Kelas,dan tipe Beras**
    - informasi yang diberikan menunjukan bahwwa akan melakukan tiga pembagian dataset yaitu : `Training`,`Validation`, dan `Test`.
    - jumlah sample yang dibagi sebesar **3000**,**2000**, dan **1250**.
    - memiliki lima kelas, terdiri dari : `arborio:0`,`basmati:1`,`ipsala:2`,`jasmine:3`,`karacadag:4`
    ''')
       
        
if __name__ == '__main__' :
        run()
        
    