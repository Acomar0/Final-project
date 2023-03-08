
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from io import StringIO
from PIL import Image

from ai_classifier0 import classifier_f



st.set_page_config(
    page_title="Assimov classificator",
    page_icon="🤖",
    layout="centered"
   
)
st.title("Asiimov")
logo = Image.open('../Logo/logo.jpg')

st.image(logo)
st.markdown("---")
image=st.file_uploader("Please upload an Image", type=["png", "jpg", "jpeg"])

if image is not None:
	st.image(image, width=600)
	classifier_f(image)
