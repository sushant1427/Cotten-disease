import streamlit as st
from PIL import Image

# Load the model
model = tf.keras.models.load_model('your_model.h5')

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

    # Make prediction
    predicted_class, confidence_score = predict(image, model, class_names)

    # Display prediction
    st.write('Predicted Class:', predicted_class)
    st.write('Confidence:', confidence_score)
