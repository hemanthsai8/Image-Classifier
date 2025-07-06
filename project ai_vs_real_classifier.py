import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model('ai_vs_real_classifier.h5')

st.title("🧠 AI vs Real Image Classifier")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")  # Ensure compatibility with model
    st.image(img, caption='📷 Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    # Updated fuzzy logic for output
    if 0.45 < confidence < 0.55:
        st.write("### 🤔 Uncertain: Low confidence in classification")
    elif confidence >= 0.5:
        st.write("### 🔥 AI Generated Image")
    else:
        st.write("### ✅ Real Image")

    st.write(f"Confidence: `{confidence * 100:.2f}%`")