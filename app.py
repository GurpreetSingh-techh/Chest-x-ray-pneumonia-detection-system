import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO


# ---------------------------------------
# Flask Application
# ---------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pneumonia_densenet121.keras"
)


# ---------------------------------------
# Load Trained Model
# ---------------------------------------

print("Loading Pneumonia Detection Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model Loaded Successfully!")


# ---------------------------------------
# Home Page
# ---------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------
# Image Prediction
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({
                "error": "Please upload an X-ray image."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "No image selected."
            }), 400

        # Read image
        image = Image.open(
            BytesIO(file.read())
        ).convert("RGB")

        # Resize image
        image = image.resize((224, 224))

        # Convert image into NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Model prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        prediction = np.array(prediction)

        # Binary sigmoid model
        if prediction.shape[-1] == 1:

            pneumonia_score = float(
                prediction.reshape(-1)[0]
            )

        # Two-class softmax model
        else:

            probabilities = prediction.reshape(-1)

            pneumonia_score = float(
                probabilities[1]
            )

        # Keep score within 0 and 1
        pneumonia_score = max(
            0.0,
            min(1.0, pneumonia_score)
        )

        normal_score = 1 - pneumonia_score

        # Classification
        if pneumonia_score >= 0.5:

            result = "PNEUMONIA"

            description = (
                "The model detected image patterns "
                "that it associates with pneumonia."
            )

        else:

            result = "NORMAL"

            description = (
                "The model classified this image "
                "as NORMAL."
            )

        return jsonify({

            "prediction": result,

            "pneumonia_score": round(
                pneumonia_score * 100, 2
            ),

            "normal_score": round(
                normal_score * 100, 2
            ),

            "description": description

        })

    except Exception as e:

        print("Prediction Error:", str(e))

        return jsonify({
            "error": "Unable to process this image. "
                     "Please try another image."
        }), 500


# ---------------------------------------
# File Size Error
# ---------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


# ---------------------------------------
# Run Application
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
