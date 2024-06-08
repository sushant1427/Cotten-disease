import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((150, 150))  # Resize image to match model's expected sizing
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array / 255.0  # Normalize pixel values

# Load the model
model = load_model('lightweight_cotton_disease_cnn_model.h5')

# Define class labels
class_names = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']

# Streamlit app
st.title('Cotton Disease Classifier')

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    # Display prediction
    st.write('Predicted Class:', predicted_class)
    st.write('Confidence:', np.max(prediction))
