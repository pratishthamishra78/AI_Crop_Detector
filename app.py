import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model and labels
model = load_model('crop_disease_model.h5')
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

st.title("🌿 Crop Disease Detection")
st.write("Upload an image file of a plant leaf (JPG, JPEG, PNG) to predict the disease.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='🖼️ Uploaded Image', use_container_width=True)

        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = model.predict(img_array)
        pred_class = labels[np.argmax(pred)]

        st.success(f"🌾 Predicted Disease: **{pred_class}**")
        st.info("⚠️ Note: This prediction is based on a trained model and may not be 100% accurate. Please consult an expert for confirmation.")
    except Exception as e:
        st.error(f"Error loading image: {e}")
