import os

import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def download_model(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner('Downloading model... please wait'):
            try:
                gdown.download(url, output_path, quiet=False)
                st.success('Model downloaded successfully!')
            except Exception as e:
                st.error(f'Failed to download model file: {e}')
                return False
    return True

# Google Drive direct download link
model_path = 'lightweight_cotton_disease_cnn_model.h5'
model_url = "https://drive.google.com/uc?export=download&id=1OUaByVCWgPa-pLcl2rNTtM3MoKSAfhqp"

if download_model(model_url, model_path):
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully")
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        st.stop()
else:
    st.stop()

def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizing the image
    return img_array

def predict_disease(image_file):
    processed_image = preprocess_image(image_file)
    st.write(f"Processed image shape: {processed_image.shape}")  # Debug statement
    prediction = model.predict(processed_image)
    st.write(f"Prediction: {prediction}")  # Debug statement
    return prediction

def main():
    st.title("Cotton Disease Prediction")
    st.write("Upload an image of a plant or leaf")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        if st.button('Classify'):
            with st.spinner("Classifying..."):
                prediction = predict_disease(uploaded_file)
                disease_class = np.argmax(prediction)
                if disease_class == 0:
                    st.write("Diseased cotton leaf")
                elif disease_class == 1:
                    st.write("Diseased cotton plant")
                elif disease_class == 2:
                    st.write("Fresh cotton leaf")
                elif disease_class == 3:
                    st.write("Fresh cotton plant")

if __name__ == '__main__':
    main()
