import os
import io
import base64

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# DOWNLOAD MODEL FROM HUGGING FACE
# --------------------------------------------------

MODEL_PATH = hf_hub_download(
    repo_id="GurpreetSingh21/x_ray_detection_system_model",
    filename="pneumonia_densenet121.keras"
)


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

    image_array = np.asarray(
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
    elif prediction.ndim == 2 and prediction.shape[-1] == 2:

        pneumonia_score = float(
            prediction[0][1]
        )

    else:

        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    pneumonia_score = float(
        np.clip(pneumonia_score, 0, 1)
    )

    normal_score = 1 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "PNEUMONIA"

        description = (
            "The model output is more consistent with "
            "the PNEUMONIA class. This is an educational "
            "AI result, not a medical diagnosis."
        )

    else:

        result = "NORMAL"

        description = (
            "The model output is more consistent with "
            "the NORMAL class. This is an educational "
            "AI result, not a medical diagnosis."
        )

    return {
        "prediction": result,
        "normal_score": round(normal_score * 100, 2),
        "pneumonia_score": round(pneumonia_score * 100, 2),
        "description": description
    }


# --------------------------------------------------
# FIND DENSENET BACKBONE FOR GRAD-CAM
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
        "No nested convolutional backbone found."
    )


# --------------------------------------------------
# GENERATE GRAD-CAM
# --------------------------------------------------

def generate_gradcam(image_array):

    backbone, last_conv_layer = find_backbone()

    x = tf.convert_to_tensor(image_array)

    # Process layers before the backbone
    for layer in model.layers:

        if layer is backbone:
            break

        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        x = layer(
            x,
            training=False
        )

    grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )

    # Get layers after the backbone
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

        if output.shape[-1] == 1:

            class_score = output[:, 0]

        elif output.shape[-1] == 2:

            class_score = output[:, 1]

        else:

            raise ValueError(
                "Unsupported model output shape."
            )

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

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

    import matplotlib

    coloured = matplotlib.colormaps["jet"](
        np.asarray(heatmap_image)
    )

    coloured = np.uint8(
        coloured[:, :, :3] * 255
    )

    coloured_image = Image.fromarray(
        coloured
    )

    original = Image.fromarray(
        np.clip(
            image_array[0],
            0,
            255
        ).astype(np.uint8)
    ).convert("RGB")

    overlay = Image.blend(
        original,
        coloured_image,
        alpha=0.40
    )

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

        image = Image.open(file.stream)

        if image.format not in ("JPEG", "PNG"):

            return jsonify({
                "error": "Please upload a JPG or PNG image."
            }), 400

        image.load()

        image = image.convert("RGB")

        image_array = preprocess_image(image)

        result = get_prediction(image_array)

        # Grad-CAM is optional.
        # If it fails, prediction will still be returned.

        try:

            result["gradcam"] = generate_gradcam(
                image_array
            )

        except Exception as grad_error:

            print("Grad-CAM Error:", repr(grad_error))

            result["gradcam"] = None

            result["gradcam_message"] = (
                "Grad-CAM is unavailable for this model architecture."
            )

        return jsonify(result)

    except UnidentifiedImageError:

        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception as error:

        print("Prediction Error:", repr(error))

        return jsonify({
            "error": "The server could not analyse this image. "
                     "Please check the Render logs."
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
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
