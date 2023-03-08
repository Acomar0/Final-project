
import numpy as np
import os
import tensorflow as tf
import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
def classifier_f():
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

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier_f():
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

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
def classifier_f():
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

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
