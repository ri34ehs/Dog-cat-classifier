import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Add this function to load the model ---
@st.cache_resource
def load_model():
    # Replace 'dog_cat_cnn_model.h5' with the actual name of your saved model file
    model = tf.keras.models.load_model('dog_cat_cnn_model.h5')
    return model

# --- Load the model at the start of the app ---
model = load_model()

st.title("Dog vs Cat Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # (Your existing code for image processing goes here)
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption='Uploaded Image.')
    
    # The prediction line will now work because 'model' is defined
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.write("## Predicted: Dog 🐶")
    else:
        st.write("## Predicted: Cat 🐱")
