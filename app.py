import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import streamlit as st 
import numpy as np
import os 


model = load_model("Image_classify.keras")


data_cat = ['Fire', 'Flood', 'earthquake']


img_height = 180
img_width = 180 

st.header("Image Classification Model - Natural Disasters")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image_load = keras_image.load_img(uploaded_file, target_size=(img_height, img_width))
    st.image(image_load, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = keras_image.img_to_array(image_load)
    img_batch = tf.expand_dims(img_array, 0)  # Create batch axis

    # Predict
    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])

    # Display results
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100

    st.write(f"🌍 The model predicts this is a **{predicted_class}**.")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
