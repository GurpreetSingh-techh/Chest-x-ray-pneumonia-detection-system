import os

# Force TensorFlow to use CPU on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
from io import BytesIO

import numpy as np
import tensorflow as tf

# Disable GPU devices
try:
    tf.config.set_visible_devices([], "GPU")
    print("TensorFlow configured to use CPU.")
except Exception as e:
    print("Device configuration:", e)

from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download


# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

# ==========================================
# HUGGING FACE CONFIGURATION
# ==========================================

HF_REPO_ID = "GurpreetSingh21/x_ray_detection_system_model"

MODEL_FILENAME = "pneumonia_densenet121.keras"

HF_TOKEN = os.environ.get("HF_TOKEN")

# ==========================================
# MODEL CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)

model = None


# ==========================================
# LOAD MODEL FROM HUGGING FACE
# ==========================================

def load_model():

    global model

    if model is not None:
        return model

    app.logger.info("Starting Hugging Face model download...")

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        token=HF_TOKEN
    )

    app.logger.info("Model downloaded: %s", model_path)

    app.logger.info("Starting TensorFlow model loading...")

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    app.logger.info("MODEL LOADED SUCCESSFULLY!")

    app.logger.info("Input shape: %s", model.input_shape)

    app.logger.info("Output shape: %s", model.output_shape)

    return model


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_file):

    image_bytes = image_file.read()

    image = Image.open(
        BytesIO(image_bytes)
    )

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==========================================
# MAKE PREDICTION
# ==========================================

def predict_image(image_array):

    loaded_model = load_model()

    prediction = loaded_model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(prediction)

    # Binary classification output

    if prediction.size == 1:

        pneumonia_score = float(
            prediction.reshape(-1)[0]
        )

        normal_score = 1.0 - pneumonia_score

    # Two-class classification output

    elif prediction.size == 2:

        scores = prediction.reshape(-1)

        normal_score = float(scores[0])

        pneumonia_score = float(scores[1])

    else:

        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    # Validate scores

    if not np.isfinite(normal_score) or not np.isfinite(pneumonia_score):
        raise ValueError("Model returned invalid prediction scores.")

    normal_score = float(
        np.clip(normal_score, 0, 1)
    )

    pneumonia_score = float(
        np.clip(pneumonia_score, 0, 1)
    )

    # Classification

    if pneumonia_score >= 0.5:

        result = "PNEUMONIA"

        description = (
            "The AI model detected image patterns "
            "associated with pneumonia."
        )

    else:

        result = "NORMAL"

        description = (
            "The AI model classified this image as normal."
        )

    return {

        "prediction": result,

        "normal_score": round(
            normal_score * 100, 2
        ),

        "pneumonia_score": round(
            pneumonia_score * 100, 2
        ),

        "description": description,

        "gradcam": None

    }


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    try:

        load_model()

        return jsonify({

            "status": "healthy",

            "model": "loaded"

        })

    except Exception as error:

        app.logger.exception(
            "Health check failed"
        )

        return jsonify({

            "status": "error",

            "message": str(error)

        }), 500


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        app.logger.info(
            "Prediction request received."
        )

        if "image" not in request.files:

            return jsonify({

                "error": "No image uploaded."

            }), 400

        image_file = request.files["image"]

        if image_file.filename == "":

            return jsonify({

                "error": "Please select an image."

            }), 400

        extension = os.path.splitext(
            image_file.filename
        )[1].lower()

        if extension not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:

            return jsonify({

                "error": "Only JPG, JPEG and PNG images are supported."

            }), 400

        # Process image

        image_array = preprocess_image(
            image_file
        )

        app.logger.info(
            "Image preprocessing completed."
        )

        # Generate prediction

        result = predict_image(
            image_array
        )

        app.logger.info(
            "Prediction completed: %s",
            result["prediction"]
        )

        return jsonify(result), 200

    except UnidentifiedImageError:

        return jsonify({

            "error": "Invalid image. Please upload a valid image."

        }), 400

    except Exception as error:

        app.logger.exception(
            "Prediction error"
        )

        return jsonify({

            "error": str(error)

        }), 500


# ==========================================
# FILE SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "error": "Image is too large. Maximum size is 10 MB."

    }), 413


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
