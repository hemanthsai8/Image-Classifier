from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('ai_vs_real_classifier.h5')

img_path = r"C:\Users\Harsha\OneDrive\ドキュメント\pro\dataset\train\ai\ai.jpg"  # change this to your actual image path
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

prediction = model.predict(x)
label = "Real" if prediction[0][0] > 0.5 else "AI"
print(f"Prediction: {label}")