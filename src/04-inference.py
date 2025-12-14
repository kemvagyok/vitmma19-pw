# Inference script
# This script runs the model on new, unseen data.
from utils import setup_logger
import config
from models import simpleCNNModel

import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image

import os
from flask import Flask, request, jsonify

logger = setup_logger()
CLASSES = ["Pronális", "Neutrális", "Szupináció"]

app = Flask(__name__)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    model = load_model(config.MODEL_SAVE_PATH)
    img = image.load_img(img_path, target_size=(config.TARGET_IMAGE_SIZE[0], config.TARGET_IMAGE_SIZE[1]))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0) / 255.0
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))

    return CLASSES[idx], float(preds[0][idx])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    label, confidence = predict_image(path)

    return jsonify({
    "label": label,
    "confidence": confidence
    })
    #logger.info(f"Prediction result: {predicts}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
