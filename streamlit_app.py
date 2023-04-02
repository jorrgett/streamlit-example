import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from werkzeug.utils import secure_filename
import cloudinary
import cloudinary.uploader
import cloudinary.api
import json

st.set_page_config(page_title="Tumor Detection", page_icon=":microscope:", layout="wide")

model = keras.models.load_model("./models/classification.h5")
classes = ['Ningún Tumor', 'Tumor Pituitario', 'Tumor Meningioma', 'Tumor Glioma']


def names(number):
    if (number == 0):
        return classes[0]
    elif (number == 1):
        return classes[1]
    elif (number == 2):
        return classes[2]
    elif (number == 3):
        return classes[3]


def upload_image(file):
    cloudinary.config(
        cloud_name="brainlypf",
        api_key="143982914773545",
        api_secret="Qt7iifjrFNj2-rFkrn9dssdYaME"
    )

    upload_data = cloudinary.uploader.upload(file)
    image_url = upload_data['secure_url']

    return image_url


def predict(image):
    dim = (150, 150)
    x = np.array(image.resize(dim))
    x = x.reshape(1, 150, 150, 3)
    answ = model.predict_on_batch(x)
    classification = np.where(answ == np.amax(answ))[1][0]
    return names(classification) + ' Detectado'


def main():
    st.title("Detección de tumores cerebrales")

    uploaded_file = st.file_uploader("Cargue una imagen de tomografía computarizada o resonancia magnética", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen cargada', use_column_width=True)
        st.write("")
        st.write("Predicción:")
        result = predict(image)
        st.write(result)


if __name__ == '__main__':
    main()
