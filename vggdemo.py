import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Load the VGG16 model
model = VGG16(weights="imagenet")

def predict(image):
    # Preprocess the image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Make predictions
    predictions = model.predict(image)
    results = decode_predictions(predictions, top=5)[0]

    return results

st.title("VGG16 Image Classification")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")
    results = predict(image)

    st.write("Results:")
    for i, (imagenet_id, label, probability) in enumerate(results):
        st.write(f"{i+1}. {label}: {probability:.2%}")


