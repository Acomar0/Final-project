import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
def classifier_f(image):
    img_height = 128
    img_width = 128
    class_names=['AI', 'Real']
    
    

    img = tf.keras.utils.load_img(
    image, target_size=(img_height, img_width)
)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model = tf.keras.models.load_model('../Models/my_model')
    


    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    

 

    st.header(
    "This image most likely belongs to {} with a :red[{:.2f}] :red[%] confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
  


