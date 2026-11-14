import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import re
import json
import base64
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import tensorflow as tf
import numpy as np

from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ClarityX | AI Chest X-Ray Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pneumonia_densenet121.keras"

HTML_PATH = BASE_DIR / "templates" / "index.html"
CSS_PATH = BASE_DIR / "static" / "style.css"
JS_PATH = BASE_DIR / "static" / "script.js"

IMAGE_SIZE = (224, 224)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    print("Loading DenseNet121 model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("MODEL LOADED SUCCESSFULLY")
    print("INPUT SHAPE:", model.input_shape)
    print("OUTPUT SHAPE:", model.output_shape)

    return model


# =========================================================
# FIND DENSENET BACKBONE
# =========================================================

def find_backbone(model):

    # First try the expected layer name.
    try:

        backbone = model.get_layer("densenet121")

        print("FOUND BACKBONE: densenet121")

        return backbone

    except ValueError:
        pass

    # Otherwise search nested models.
    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            try:

                layer.get_layer(
                    "conv5_block16_concat"
                )

                print(
                    "FOUND BACKBONE:",
                    layer.name
                )

                return layer

            except ValueError:
                continue

    raise ValueError(
        "DenseNet121 backbone not found. "
        "Check model.summary() and nested layer names."
    )


# =========================================================
# BUILD GRAD-CAM MODEL
# =========================================================

@st.cache_resource
def build_gradcam_model():

    model = load_model()

    backbone = find_backbone(model)

    # Final DenseNet121 convolutional layer.
    conv_layer = backbone.get_layer(
        "conv5_block16_concat"
    )

    print(
        "GRAD-CAM CONVOLUTIONAL LAYER:",
        conv_layer.name
    )

    # Feature extractor.
    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            conv_layer.output,
            backbone.output
        ],
        name="gradcam_feature_extractor"
    )

    # Find backbone in the original model.
    backbone_index = model.layers.index(
        backbone
    )

    # Classification head after DenseNet121.
    classifier_layers = model.layers[
        backbone_index + 1:
    ]

    if not classifier_layers:

        raise ValueError(
            "Classification head was not found."
        )

    print(
        "CLASSIFICATION HEAD:",
        [layer.name for layer in classifier_layers]
    )

    print("GRAD-CAM MODEL READY")

    return feature_extractor, classifier_layers


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    original_image = image.copy()

    resized_image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Must match DenseNet121 training preprocessing.
    image_array = (
        tf.keras.applications.densenet.preprocess_input(
            image_array
        )
    )

    return image_array, original_image


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(
    image_array,
    predicted_class
):

    print("STARTING GRAD-CAM GENERATION")

    feature_extractor, classifier_layers = (
        build_gradcam_model()
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        # Get convolutional feature maps.
        conv_features, backbone_output = (
            feature_extractor(
                image_tensor,
                training=False
            )
        )

        x = backbone_output

        # Forward pass through classification head.
        for layer in classifier_layers:

            x = layer(
                x,
                training=False
            )

        # Model has one sigmoid output.
        pneumonia_score = x[:, 0]

        if predicted_class == "PNEUMONIA":

            target_score = pneumonia_score

        else:

            target_score = 1.0 - pneumonia_score

    # Gradients with respect to convolutional features.
    gradients = tape.gradient(
        target_score,
        conv_features
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None. "
            "Check the model architecture."
        )

    # Average gradients spatially.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension.
    channel_weights = channel_weights[0]

    conv_features = conv_features[0]

    # Weighted sum of feature maps.
    heatmap = tf.reduce_sum(
        conv_features * channel_weights,
        axis=-1
    )

    # Keep positive activations.
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize.
    maximum = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        maximum
    )

    heatmap = heatmap.numpy()

    if not np.isfinite(heatmap).all():

        raise RuntimeError(
            "Grad-CAM contains invalid values."
        )

    if float(np.max(heatmap)) <= 0:

        raise RuntimeError(
            "Grad-CAM is empty. No positive activation found."
        )

    print("GRAD-CAM GENERATED SUCCESSFULLY")

    return heatmap


# =========================================================
# CREATE HEATMAP OVERLAY
# =========================================================

def create_gradcam_image(
    original_image,
    heatmap
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Resize heatmap to original X-ray size.
    heatmap_image = Image.fromarray(
        np.uint8(
            np.clip(heatmap, 0, 1) * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.asarray(heatmap_image, dtype=np.float32)
        / 255.0
    )

    # JET colour map.
    jet = plt.colormaps["jet"]

    colored_heatmap = jet(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    # Original image.
    original_array = np.asarray(
        original_image.convert("RGB")
    )

    # Blend.
    overlay = (
        0.65 * original_array.astype(np.float32)
        +
        0.35 * colored_heatmap.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    overlay_image = Image.fromarray(
        overlay
    )

    buffer = io.BytesIO()

    overlay_image.save(
        buffer,
        format="PNG",
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    print("GRAD-CAM IMAGE CREATED")

    return encoded


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image):

    print("\nNEW IMAGE REQUEST")

    image_array, original_image = preprocess_image(
        image
    )

    model = load_model()

    print("RUNNING MODEL PREDICTION")

    raw_prediction = model(
        image_array,
        training=False
    ).numpy()

    pneumonia_probability = float(
        raw_prediction.reshape(-1)[0]
    )

    pneumonia_probability = float(
        np.clip(
            pneumonia_probability,
            0.0,
            1.0
        )
    )

    normal_probability = (
        1.0 - pneumonia_probability
    )

    if pneumonia_probability >= 0.5:

        predicted_class = "PNEUMONIA"

    else:

        predicted_class = "NORMAL"

    print("PREDICTION:", predicted_class)

    normal_score = round(
        normal_probability * 100,
        2
    )

    pneumonia_score = round(
        pneumonia_probability * 100,
        2
    )

    if predicted_class == "PNEUMONIA":

        description = (
            "The model identified patterns associated "
            "with the Pneumonia class."
        )

    else:

        description = (
            "The model classified this image as Normal."
        )

    # Always try Grad-CAM.
    gradcam_base64 = None

    gradcam_message = None

    try:

        heatmap = generate_gradcam(
            image_array,
            predicted_class
        )

        gradcam_base64 = create_gradcam_image(
            original_image,
            heatmap
        )

        gradcam_message = (
            "Grad-CAM generated successfully."
        )

        print("GRAD-CAM AVAILABLE: TRUE")

    except Exception as error:

        gradcam_message = str(error)

        print("GRAD-CAM ERROR:", repr(error))

    response = {

        "prediction": predicted_class,

        "normal_score": normal_score,

        "pneumonia_score": pneumonia_score,

        "description": description,

        "gradcam": gradcam_base64,

        "gradcam_message": gradcam_message

    }

    print(
        "GRAD-CAM BASE64 LENGTH:",
        len(gradcam_base64) if gradcam_base64 else 0
    )

    return response


# =========================================================
# LOAD YOUR EXISTING FRONTEND
# =========================================================

def load_frontend():

    required_files = [
        HTML_PATH,
        CSS_PATH,
        JS_PATH
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Frontend file missing: {file_path}"
            )

    html = HTML_PATH.read_text(
        encoding="utf-8"
    )

    css = CSS_PATH.read_text(
        encoding="utf-8"
    )

    js = JS_PATH.read_text(
        encoding="utf-8"
    )

    # Remove original Flask stylesheet reference.
    html = re.sub(
        r'<link[^>]*href="\{\{\s*url_for\([\'"]static[\'"],\s*filename=[\'"]style\.css[\'"]\)\s*\}\}"[^>]*>',
        "",
        html
    )

    # Remove original Flask JavaScript reference.
    html = re.sub(
        r'<script[^>]*src="\{\{\s*url_for\([\'"]static[\'"],\s*filename=[\'"]script\.js[\'"],\s*v=2\)\s*\}\}"[^>]*>\s*</script>',
        "",
        html
    )

    # Insert CSS without modifying the CSS file.
    html = html.replace(
        "</head>",
        "<style>\n" + css + "\n</style>\n</head>"
    )

    # Insert your original JavaScript without modifying it.
    html = html.replace(
        "</body>",
        "<script>\n" + js + "\n</script>\n</body>"
    )

    return html


# =========================================================
# STREAMLIT COMPONENT BRIDGE
# =========================================================

BRIDGE_DIR = BASE_DIR / ".clarityx_component"

BRIDGE_DIR.mkdir(
    exist_ok=True
)

BRIDGE_FILE = BRIDGE_DIR / "index.html"


def create_component_html():

    frontend = load_frontend()

    bridge = r"""

<script>

let pendingRequest = null;


// Streamlit component initialization.
window.parent.postMessage({

    isStreamlitMessage: true,

    type: "streamlit:componentReady",

    apiVersion: 1

}, "*");


window.parent.postMessage({

    isStreamlitMessage: true,

    type: "streamlit:setFrameHeight",

    height: 2200

}, "*");


// Receive prediction result from Python.
window.addEventListener("message", function(event) {

    const message = event.data;

    if (!message || !message.type) {
        return;
    }

    if (message.type === "streamlit:render") {

        const args = message.args || {};

        const response = args.response;

        if (
            response &&
            pendingRequest &&
            response.request_id === pendingRequest.request_id
        ) {

            const result = response.data;

            pendingRequest.resolve(

                new Response(
                    JSON.stringify(result),
                    {
                        status: 200,
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                )

            );

            pendingRequest = null;

        }

    }

});


// Intercept the original JavaScript fetch request.
const originalFetch = window.fetch.bind(window);

window.fetch = async function(input, init = {}) {

    const requestUrl = (
        typeof input === "string"
        ? input
        : input.url
    );

    const pathname = new URL(
        requestUrl,
        window.location.href
    ).pathname;

    if (pathname !== "/predict") {

        return originalFetch(input, init);

    }

    const formData = init.body;

    if (!(formData instanceof FormData)) {

        return originalFetch(input, init);

    }

    const file = formData.get("image");

    if (!(file instanceof File)) {

        return new Response(
            JSON.stringify({
                error: "No image file received."
            }),
            {
                status: 400,
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

    }

    const requestId = (
        window.crypto &&
        window.crypto.randomUUID
    )
        ? window.crypto.randomUUID()
        : String(Date.now()) + Math.random();


    return new Promise(function(resolve, reject) {

        pendingRequest = {

            request_id: requestId,

            resolve: resolve,

            reject: reject

        };


        const reader = new FileReader();


        reader.onload = function() {

            const base64Image = reader.result.split(",")[1];


            // Send image to Streamlit Python.
            window.parent.postMessage({

                isStreamlitMessage: true,

                type: "streamlit:setComponentValue",

                dataType: "json",

                value: {

                    request_id: requestId,

                    filename: file.name,

                    image: base64Image

                }

            }, "*");

        };


        reader.onerror = function() {

            pendingRequest = null;

            reject(
                new Error("Unable to read selected image.")
            );

        };


        reader.readAsDataURL(file);

    });

};

</script>

"""

    BRIDGE_FILE.write_text(
        frontend + bridge,
        encoding="utf-8"
    )


# =========================================================
# DECLARE COMPONENT
# =========================================================

create_component_html()

clarityx_component = components.declare_component(
    "clarityx_frontend",
    path=str(BRIDGE_DIR)
)


# =========================================================
# STREAMLIT BACKEND
# =========================================================

if "clarityx_response" not in st.session_state:

    st.session_state.clarityx_response = None


if "clarityx_processed" not in st.session_state:

    st.session_state.clarityx_processed = set()


# Render complete original webpage.
component_value = clarityx_component(
    response=st.session_state.clarityx_response,
    key="clarityx_frontend_component"
)


# Receive image request.
if isinstance(component_value, dict):

    request_id = component_value.get(
        "request_id"
    )

    if (
        request_id
        and request_id not in st.session_state.clarityx_processed
    ):

        st.session_state.clarityx_processed.add(
            request_id
        )

        try:

            image_bytes = base64.b64decode(
                component_value["image"],
                validate=True
            )

            if len(image_bytes) > 10 * 1024 * 1024:

                raise ValueError(
                    "Image must be smaller than 10 MB."
                )

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image.verify()

            image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            result = predict_image(image)

        except Exception as error:

            print("REQUEST FAILED:", repr(error))

            result = {
                "error": str(error)
            }

        st.session_state.clarityx_response = {

            "request_id": request_id,

            "data": result

        }

        # Send result back to the frontend.
        st.rerun()
