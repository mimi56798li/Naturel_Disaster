import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st 
import numpy as np
model = load_model("C:/Users/nakou/OneDrive/Desktop/NaturelDisaster/Image_classify.keras")
data_cat =['Fire', 'Flood', 'earthquake']
img_height =180
img_width =180 
image = "C:/Users/nakou/OneDrive/Desktop/NaturelDisaster/Model/earthquake.jpg"

image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)
predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

st.image(image)
st.write("Naturel disaster in image is "+ data_cat[np.argmax(score)])
st.write("with acuracy of "+ str(np.max(score)*100))
