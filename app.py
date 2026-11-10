import os
import io
import base64

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# FLASK APP CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="GurpreetSingh21/x_ray_detection_system_model",
    filename="pneumonia_densenet121.keras"
)

IMAGE_SIZE = (224, 224)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading Pneumonia Detection Model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

def get_prediction(image_array):

    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(prediction)

    # Binary sigmoid output
    if prediction.size == 1:

        pneumonia_score = float(
            prediction.reshape(-1)[0]
        )

    # Two-class softmax output
    elif prediction.shape[-1] == 2:

        pneumonia_score = float(
            prediction.reshape(-1, 2)[0][1]
        )

    else:

        raise ValueError(
            "Unexpected model output shape."
        )

    pneumonia_score = float(
        np.clip(pneumonia_score, 0, 1)
    )

    normal_score = 1 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "PNEUMONIA"

        description = (
            "The model output is more consistent "
            "with the PNEUMONIA class."
        )

    else:

        result = "NORMAL"

        description = (
            "The model output is more consistent "
            "with the NORMAL class."
        )

    return {
        "prediction": result,
        "normal_score": round(normal_score * 100, 2),
        "pneumonia_score": round(pneumonia_score * 100, 2),
        "description": description
    }


# --------------------------------------------------
# FIND DENSENET BASE MODEL FOR GRAD-CAM
# --------------------------------------------------

def find_backbone():

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            conv_layers = [
                item for item in layer.layers
                if isinstance(item, tf.keras.layers.Conv2D)
            ]

            if conv_layers:

                return layer, conv_layers[-1]

    raise ValueError(
        "Could not find a nested convolutional backbone."
    )


# --------------------------------------------------
# GRAD-CAM HEATMAP
# --------------------------------------------------

def generate_gradcam(image_array):

    backbone, last_conv_layer = find_backbone()

    # Pass the image through layers before DenseNet.
    x = tf.convert_to_tensor(image_array)

    for layer in model.layers:

        if layer is backbone:
            break

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(x, training=False)

    # Build a model that returns:
    # 1. Last convolutional feature maps
    # 2. Backbone output

    grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )

    # Pass backbone output through remaining layers.
    following_layers = []

    found_backbone = False

    for layer in model.layers:

        if layer is backbone:

            found_backbone = True
            continue

        if found_backbone:

            following_layers.append(layer)

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = grad_model(
            x,
            training=False
        )

        output = backbone_output

        for layer in following_layers:

            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            output = layer(
                output,
                training=False
            )

        # Binary sigmoid or two-class softmax
        if output.shape[-1] == 1:

            class_score = output[:, 0]

        else:

            class_score = output[:, 1]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

    # Average gradients across spatial dimensions.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        maximum + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # Convert heatmap to an image.
    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_image = Image.fromarray(
        heatmap
    )

    heatmap_image = heatmap_image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # Apply a colour map.
    import matplotlib

    coloured = matplotlib.colormaps["jet"](
        np.array(heatmap_image)
    )

    coloured = np.uint8(
        coloured[:, :, :3] * 255
    )

    coloured_image = Image.fromarray(
        coloured
    )

    # Original X-ray image
    original = Image.fromarray(
        image_array[0].astype(np.uint8)
    ).convert("RGB")

    # Blend heatmap with original image.
    overlay = Image.blend(
        original,
        coloured_image,
        alpha=0.40
    )

    # Convert overlay into Base64.
    buffer = io.BytesIO()

    overlay.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded_image


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# PREDICTION API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "Please upload a chest X-ray image."
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error": "No image selected."
            }), 400

        # Validate image format.
        image = Image.open(file.stream)

        if image.format not in ["JPEG", "PNG"]:

            return jsonify({
                "error": "Please upload a JPG or PNG image."
            }), 400

        image.load()

        image = image.convert("RGB")

        # Preprocess image.
        image_array = preprocess_image(image)

        # Prediction.
        result = get_prediction(image_array)

        # Grad-CAM.
        try:

            gradcam_image = generate_gradcam(
                image_array
            )

            result["gradcam"] = gradcam_image

        except Exception as grad_error:

            print("Grad-CAM Error:", grad_error)

            result["gradcam"] = None

            result["gradcam_message"] = (
                "Grad-CAM could not be generated for this model."
            )

        return jsonify(result)

    except UnidentifiedImageError:

        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception as error:

        print("Prediction Error:", error)

        return jsonify({
            "error": "Something went wrong while analysing the image."
        }), 500


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
