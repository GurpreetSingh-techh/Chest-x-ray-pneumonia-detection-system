```python
import os
import re
import io
import base64
import uuid
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np

from PIL import Image, UnidentifiedImageError


# =====================================================
# STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ClarityX - Chest X-ray AI",
    page_icon="🩻",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pneumonia_densenet121.keras"

HTML_PATH = BASE_DIR / "templates" / "index.html"
CSS_PATH = BASE_DIR / "static" / "style.css"
JS_PATH = BASE_DIR / "static" / "script.js"

IMAGE_SIZE = (224, 224)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    print("LOADING PNEUMONIA MODEL...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("MODEL LOADED SUCCESSFULLY!")
    print("MODEL INPUT:", model.input_shape)
    print("MODEL OUTPUT:", model.output_shape)

    return model


# =====================================================
# BUILD GRAD-CAM MODEL
# =====================================================

@st.cache_resource
def build_gradcam_components():

    model = load_model()

    print("BUILDING GRAD-CAM MODEL...")

    # Find DenseNet121 backbone.
    backbone = None

    try:
        backbone = model.get_layer("densenet121")
    except ValueError:
        pass

    # Fallback: search nested models.
    if backbone is None:

        candidates = [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.Model)
        ]

        for candidate in candidates:

            try:
                candidate.get_layer("conv5_block16_concat")
                backbone = candidate
                break

            except ValueError:
                continue

    if backbone is None:
        raise ValueError(
            "DenseNet121 backbone was not found."
        )

    # Find the final convolutional feature layer.
    last_conv_layer = backbone.get_layer(
        "conv5_block16_concat"
    )

    # Extract convolutional features and backbone output.
    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )

    # Find the backbone's position in the main model.
    backbone_index = model.layers.index(backbone)

    # Remaining classification layers.
    classification_layers = model.layers[
        backbone_index + 1:
    ]

    print("GRAD-CAM LAYER:", last_conv_layer.name)
    print("GRAD-CAM MODEL CREATED SUCCESSFULLY!")

    return (
        feature_extractor,
        classification_layers
    )


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image = image.convert("RGB")

    original_image = image.copy()

    resized_image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # DenseNet121 preprocessing.
    image_array = (
        tf.keras.applications.densenet.preprocess_input(
            image_array
        )
    )

    return image_array, original_image


# =====================================================
# GRAD-CAM GENERATION
# =====================================================

def generate_gradcam(image_array, predicted_class):

    model = load_model()

    feature_extractor, classification_layers = (
        build_gradcam_components()
    )

    print("GENERATING GRAD-CAM...")

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        # Extract feature maps.
        conv_outputs, backbone_output = (
            feature_extractor(
                image_tensor,
                training=False
            )
        )

        x = backbone_output

        # Pass through the original classifier layers.
        for layer in classification_layers:

            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            x = layer(
                x,
                training=False
            )

        predictions = x

        # Binary sigmoid model:
        # 0 = NORMAL
        # 1 = PNEUMONIA

        if predicted_class == "PNEUMONIA":

            class_score = predictions[:, 0]

        else:

            class_score = 1.0 - predictions[:, 0]

    # Calculate gradients.
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Gradients could not be calculated."
        )

    # Global average pooling of gradients.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension.
    weights = weights[0]

    conv_outputs = conv_outputs[0]

    # Weighted combination of feature maps.
    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    # ReLU removes negative activations.
    cam = tf.maximum(
        cam,
        0
    )

    # Normalize heatmap.
    max_value = tf.reduce_max(cam)

    cam = tf.math.divide_no_nan(
        cam,
        max_value
    )

    heatmap = cam.numpy()

    print("GRAD-CAM GENERATED SUCCESSFULLY!")

    return heatmap


# =====================================================
# CREATE GRAD-CAM OVERLAY
# =====================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    print("CREATING GRAD-CAM OVERLAY...")

    # Resize heatmap to original image dimensions.
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.array(heatmap_image).astype(np.float32)
        / 255.0
    )

    # Apply JET color map.
    colored_heatmap = plt.colormaps["jet"](
        heatmap_array
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    # Original X-ray.
    original_array = np.array(
        original_image.convert("RGB")
    )

    # Blend original image and heatmap.
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

    # Convert image to Base64.
    buffer = io.BytesIO()

    overlay_image.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    print("GRAD-CAM OVERLAY CREATED!")

    return encoded_image


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_image(uploaded_image):

    print("\nNEW REQUEST RECEIVED")

    try:

        # Preprocessing.
        image_array, original_image = (
            preprocess_image(uploaded_image)
        )

        print("IMAGE PREPROCESSING COMPLETED")

        # Model prediction.
        model = load_model()

        print("PREDICTION STARTED")

        prediction = model.predict(
            image_array,
            verbose=0
        )

        pneumonia_probability = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        print(
            "PNEUMONIA PROBABILITY:",
            pneumonia_probability
        )

        # Classification.
        if pneumonia_probability >= 0.5:

            predicted_class = "PNEUMONIA"

        else:

            predicted_class = "NORMAL"

        normal_score = (
            1.0 - pneumonia_probability
        ) * 100

        pneumonia_score = (
            pneumonia_probability * 100
        )

        print("PREDICTION:", predicted_class)

        # Description.
        if predicted_class == "PNEUMONIA":

            description = (
                "The AI model detected patterns "
                "associated with pneumonia."
            )

        else:

            description = (
                "The AI model classified this image "
                "as normal."
            )

        # Generate Grad-CAM.
        gradcam_base64 = None

        gradcam_message = None

        try:

            heatmap = generate_gradcam(
                image_array,
                predicted_class
            )

            gradcam_base64 = create_gradcam_overlay(
                original_image,
                heatmap
            )

            gradcam_message = (
                "Grad-CAM visualization generated successfully."
            )

            print("GRAD-CAM AVAILABLE: True")

        except Exception as grad_error:

            print("GRAD-CAM GENERATION FAILED!")
            print(str(grad_error))

            gradcam_message = (
                "Grad-CAM could not be generated: "
                + str(grad_error)
            )

        # Response to existing JavaScript.
        response = {

            "prediction": predicted_class,

            "normal_score": round(
                normal_score,
                2
            ),

            "pneumonia_score": round(
                pneumonia_score,
                2
            ),

            "description": description,

            "gradcam": gradcam_base64,

            "gradcam_message": gradcam_message

        }

        print("SENDING RESPONSE TO FRONTEND")

        return response

    except Exception as error:

        print("PREDICTION ERROR:", str(error))

        return {
            "error": str(error)
        }


# =====================================================
# READ ORIGINAL FRONTEND FILES
# =====================================================

def read_frontend():

    html = HTML_PATH.read_text(
        encoding="utf-8"
    )

    css = CSS_PATH.read_text(
        encoding="utf-8"
    )

    js = JS_PATH.read_text(
        encoding="utf-8"
    )

    # Replace Flask static references.
    html = re.sub(
        r"""<link[^>]*href=["']\{\{\s*url_for\(['"]static['"],\s*filename=['"]style\.css['"]\)\s*\}\}["'][^>]*>""",
        "",
        html
    )

    html = re.sub(
        r"""<script[^>]*src=["']\{\{\s*url_for\(['"]static['"],\s*filename=['"]script\.js['"][^)]*\)\s*\}\}["'][^>]*>\s*</script>""",
        "",
        html
    )

    # Insert original CSS and JavaScript.
    html = html.replace(
        "</head>",
        "<style>" + css + "</style></head>"
    )

    html = html.replace(
        "</body>",
        "<script>" + js + "</script></body>"
    )

    return html


# =====================================================
# STREAMLIT CUSTOM COMPONENT
# =====================================================

BRIDGE_DIR = BASE_DIR / ".xray_bridge"

BRIDGE_DIR.mkdir(
    exist_ok=True
)

BRIDGE_HTML = BRIDGE_DIR / "index.html"


def create_bridge():

    frontend_html = read_frontend()

    bridge_script = r"""
<script>

let pendingRequest = null;
let lastResponseId = null;

function sendMessage(type, data = {}) {

    window.parent.postMessage({

        isStreamlitMessage: true,
        type: type,
        ...data

    }, "*");

}

sendMessage("streamlit:componentReady", {
    apiVersion: 1
});

sendMessage("streamlit:setFrameHeight", {
    height: 1000
});


// Receive Python responses.
window.addEventListener("message", function(event) {

    const message = event.data;

    if (!message || !message.type) {
        return;
    }

    if (message.type === "streamlit:render") {

        const response = message.args.response;

        if (
            response &&
            pendingRequest &&
            response.request_id === pendingRequest.id
        ) {

            pendingRequest.resolve(

                new Response(
                    JSON.stringify(response.data),
                    {
                        status: 200,
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                )

            );

            pendingRequest = null;

            lastResponseId = response.request_id;

        }

    }

});


// Preserve the original frontend JavaScript.
const originalFetch = window.fetch.bind(window);

window.fetch = async function(input, init = {}) {

    const url = typeof input === "string"
        ? input
        : input.url;

    if (
        url &&
        new URL(url, window.location.href).pathname === "/predict"
    ) {

        const formData = init.body;

        if (!(formData instanceof FormData)) {
            return originalFetch(input, init);
        }

        const file = formData.get("image");

        if (!(file instanceof File)) {

            return new Response(
                JSON.stringify({
                    error: "No image was uploaded."
                }),
                {
                    status: 400,
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );

        }

        const requestId = crypto.randomUUID();

        return new Promise((resolve) => {

            pendingRequest = {
                id: requestId,
                resolve: resolve
            };

            const reader = new FileReader();

            reader.onload = function() {

                const base64Data = reader.result.split(",")[1];

                sendMessage(
                    "streamlit:setComponentValue",
                    {
                        value: {
                            request_id: requestId,
                            filename: file.name,
                            image: base64Data
                        }
                    }
                );

            };

            reader.onerror = function() {

                pendingRequest = null;

                resolve(
                    new Response(
                        JSON.stringify({
                            error: "Could not read uploaded image."
                        }),
                        {
                            status: 400,
                            headers: {
                                "Content-Type": "application/json"
                            }
                        }
                    )
                );

            };

            reader.readAsDataURL(file);

        });

    }

    return originalFetch(input, init);

};

</script>
"""

    full_html = (
        frontend_html
        + bridge_script
    )

    BRIDGE_HTML.write_text(
        full_html,
        encoding="utf-8"
    )


# =====================================================
# DECLARE COMPONENT
# =====================================================

create_bridge()

xray_component = components.declare_component(
    "xray_bridge",
    path=str(BRIDGE_DIR)
)


# =====================================================
# STREAMLIT APPLICATION
# =====================================================

st.title("ClarityX - Chest X-ray AI")

st.caption(
    "AI-powered chest X-ray classification "
    "with Grad-CAM visualization."
)


if "bridge_response" not in st.session_state:

    st.session_state.bridge_response = None


if "processed_requests" not in st.session_state:

    st.session_state.processed_requests = set()


if "bridge_error" not in st.session_state:

    st.session_state.bridge_error = None


# Receive requests from frontend.
component_value = xray_component(
    response=st.session_state.bridge_response,
    key="xray_frontend"
)


if component_value is not None:

    request_id = component_value.get(
        "request_id"
    )

    if (
        request_id
        and request_id not in st.session_state.processed_requests
    ):

        st.session_state.processed_requests.add(
            request_id
        )

        try:

            image_bytes = base64.b64decode(
                component_value["image"]
            )

            uploaded_image = Image.open(
                io.BytesIO(image_bytes)
            )

            uploaded_image.verify()

            uploaded_image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

            response = predict_image(
                uploaded_image
            )

        except Exception as error:

            response = {
                "error": str(error)
            }

        st.session_state.bridge_response = {

            "request_id": request_id,

            "data": response

        }

        st.rerun()
```
