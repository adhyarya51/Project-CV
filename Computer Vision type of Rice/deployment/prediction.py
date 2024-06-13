# import libraries
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# load the files !
model_tl_vgg19 = load_model('model_tl_vgg19.h5')

def run():
    # judul
    st.title('Prediksi Klasifikasi Beras')

# def predict_art():
    class_labels = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag', 'Unknown']
    uploaded = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])

    if uploaded is not None:
        # Load the image
        img = tf.keras.utils.load_img(uploaded, target_size=(224, 224))  # resize to model's input size
        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        x = tf.keras.utils.img_to_array(img)  # convert image to array
        x = np.expand_dims(x, axis=0)

        # Predict the class of the image
        predictions = model_tl_vgg19.predict(x)
        
        # Get the index of the highest predicted score
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]

        # Display the prediction
        st.write('Prediction:', predicted_class_label)


if __name__ == '__main__':
    run()
    