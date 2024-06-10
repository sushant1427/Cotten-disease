import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os

# Custom CSS to add a background image
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1599820633726-c568d7a7b060");
background-size: cover;
background-position: center;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to download the model file if not present
def download_model(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner('Downloading model... Please wait...'):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success('Model downloaded successfully!')
            else:
                st.error(f'Failed to download the model file. Status code: {r.status_code}. Check the URL or network.')
                return False
    return True

# Set the path and URL for the model
model_path = 'cotton_disease_model.h5'
model_url = 'https://drive.google.com/uc?export=download&id=19YWiGmG73dYLclzTggVxsfc0hxLrWGH_'

# Ensure model is downloaded
if download_model(model_url, model_path):
    # Load the trained model
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.stop()

# Function to preprocess the image
def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_disease(image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Cotton Plant Diseases Classification")
    st.write("Upload an image of a cotton plant or leaf to classify its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")

        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = predict_disease(uploaded_file)
                disease_class = np.argmax(prediction)
                if disease_class == 0:
                    st.write("Prediction: Diseased cotton leaf")
                elif disease_class == 1:
                    st.write("Prediction: Diseased cotton plant")
                elif disease_class == 2:
                    st.write("Prediction: Fresh cotton leaf")
                elif disease_class == 3:
                    st.write("Prediction: Fresh cotton plant")

if __name__ == '__main__':
    main()
