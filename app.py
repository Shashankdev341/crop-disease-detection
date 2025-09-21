from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS  # solves CORS issues

app = Flask(__name__)
CORS(app)

# -----------------------------
# Path to model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "leaf_disease_model.h5")

# -----------------------------
# Load trained model
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Dynamically get class names from dataset folder
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "datasheet", "train")
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("Detected classes:", class_names)

# -----------------------------
# Prescriptions (map per class)
# -----------------------------
default_prescriptions = {
    "Healthy": ["No treatment needed"],
    "Early Blight": ["Use fungicide containing chlorothalonil", "Remove affected leaves"],
    "Late Blight": ["Apply copper-based fungicide", "Avoid overhead watering"],
    "Powdery Mildew": ["Use sulfur spray", "Ensure proper air circulation"],
    "Leaf Spot": ["Apply neem oil", "Keep foliage dry"],
    "Nutrient Deficiency": ["Use balanced fertilizer", "Add micronutrients if needed"]
}

def get_prescription(disease):
    return default_prescriptions.get(disease, ["Consult expert"])

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "Backend is working!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        # Load image
        img = image.load_img(file, target_size=(128,128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])

        disease_name = class_names[class_index]
        response = {
            "class_name": disease_name,
            "probability": confidence,
            "prescriptions": get_prescription(disease_name)
        }

        return jsonify(response)

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
