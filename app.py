import os
import re
import base64
import io
import uuid

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError


# ==========================================================
# STREAMLIT CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="ClarityX - X-Ray Detection",
    page_icon="🩻",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")
CSS_PATH = os.path.join(BASE_DIR, "static", "style.css")
JS_PATH = os.path.join(BASE_DIR, "static", "script.js")

MODEL_PATHS = [
    os.path.join(BASE_DIR, "pneumonia_densenet121.keras"),
    os.path.join(BASE_DIR, "models", "pneumonia_densenet121.keras")
]

BRIDGE_DIR = os.path.join(BASE_DIR, ".xray_bridge")


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model_path = next(
        (path for path in MODEL_PATHS if os.path.exists(path)),
        None
    )

    if model_path is None:
        raise FileNotFoundError(
            "Model not found. Put pneumonia_densenet121.keras "
            "in the project root or inside the models folder."
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


try:
    model = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# IMPORTANT:
# This preprocessing assumes your trained model expects
# RGB images resized to 224x224 with pixel values 0-255.
# It must match the preprocessing used during model training.


# ==========================================================
# MODEL PREDICTION
# ==========================================================

def predict_image(image_bytes):

    image_array = preprocess_image(image_bytes)

    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.array(prediction).flatten()

    if len(prediction) == 1:

        pneumonia_score = float(prediction[0])

    elif len(prediction) == 2:

        pneumonia_score = float(prediction[1])

    else:

        raise ValueError(
            "Unexpected model output. "
            "Expected one sigmoid output or two softmax outputs."
        )

    pneumonia_score = float(
        np.clip(pneumonia_score, 0, 1)
    )

    normal_score = 1 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "Pneumonia Detected"

        description = (
            "The model detected patterns associated with "
            "pneumonia in this X-ray image."
        )

    else:

        result = "Normal"

        description = (
            "The model classified this X-ray image as normal."
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

        "gradcam": None,

        "gradcam_message": (
            "Grad-CAM visualization is not configured "
            "in this Streamlit backend."
        )
    }


# ==========================================================
# CREATE STREAMLIT FRONTEND BRIDGE
# ==========================================================

def create_bridge():

    os.makedirs(
        BRIDGE_DIR,
        exist_ok=True
    )

    with open(
        HTML_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        html = file.read()

    with open(
        CSS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        css = file.read()

    with open(
        JS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        javascript = file.read()

    # ------------------------------------------------------
    # BRIDGE JAVASCRIPT
    # ------------------------------------------------------

    bridge_javascript = r"""
    <script>

    // Send messages to Streamlit

    function sendToStreamlit(type, data) {

        window.parent.postMessage({

            isStreamlitMessage: true,

            type: type,

            ...data

        }, "*");

    }


    // Notify Streamlit that the component is ready

    sendToStreamlit(
        "streamlit:componentReady",
        { apiVersion: 1 }
    );


    // Keep track of pending prediction requests

    const pendingRequests = new Map();


    // Receive prediction results from Python

    window.addEventListener("message", function(event) {

        const message = event.data;

        if (!message || !message.isStreamlitMessage) {
            return;
        }

        if (message.type === "streamlit:render") {

            const args = message.args || {};

            const response = args.response;

            if (response && response.id) {

                const pending = pendingRequests.get(
                    response.id
                );

                if (pending) {

                    pendingRequests.delete(
                        response.id
                    );

                    pending.resolve(
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

                }

            }

        }

    });


    // ------------------------------------------------------
    // INTERCEPT THE ORIGINAL FETCH REQUEST
    // ------------------------------------------------------

    const originalFetch = window.fetch.bind(window);

    window.fetch = async function(input, init = {}) {

        const url = typeof input === "string"
            ? input
            : input.url;

        if (
            url === "/predict" &&
            init.body instanceof FormData
        ) {

            const file = init.body.get("image");

            if (!file) {

                return new Response(
                    JSON.stringify({
                        error: "No image uploaded."
                    }),
                    {
                        status: 400,
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }
                );

            }

            return new Promise((resolve) => {

                const reader = new FileReader();

                reader.onload = function() {

                    const base64Image =
                        reader.result.split(",")[1];

                    const requestId =
                        Date.now().toString() +
                        Math.random().toString(36).substring(2);


                    pendingRequests.set(
                        requestId,
                        {
                            resolve: resolve
                        }
                    );


                    sendToStreamlit(
                        "streamlit:setComponentValue",
                        {
                            dataType: "json",

                            value: {

                                id: requestId,

                                filename: file.name,

                                image: base64Image

                            }

                        }
                    );

                };


                reader.readAsDataURL(file);

            });

        }


        return originalFetch(input, init);

    };


    // ------------------------------------------------------
    // SET COMPONENT HEIGHT
    // ------------------------------------------------------

    function updateHeight() {

        const height = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight
        );

        sendToStreamlit(
            "streamlit:setFrameHeight",
            { height: height }
        );

    }

    window.addEventListener(
        "load",
        updateHeight
    );

    window.addEventListener(
        "resize",
        updateHeight
    );

    new ResizeObserver(
        updateHeight
    ).observe(document.body);

    </script>
    """


    # ------------------------------------------------------
    # INSERT ORIGINAL CSS
    # ------------------------------------------------------

    style_tag = f"""
    <style>
    {css}
    </style>
    """

    html = re.sub(
        r'<link[^>]+style\.css[^>]*>',
        lambda match: style_tag,
        html,
        flags=re.IGNORECASE
    )


    # ------------------------------------------------------
    # REPLACE ORIGINAL JAVASCRIPT FILE REFERENCE
    # ------------------------------------------------------

    script_tag = f"""
    {bridge_javascript}

    <script>
    {javascript}
    </script>
    """

    html = re.sub(
        r'<script[^>]+script\.js[^>]*>\s*</script>',
        lambda match: script_tag,
        html,
        flags=re.IGNORECASE
    )


    # ------------------------------------------------------
    # SAVE GENERATED COMPONENT HTML
    # ------------------------------------------------------

    component_html_path = os.path.join(
        BRIDGE_DIR,
        "index.html"
    )

    with open(
        component_html_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(html)

    return BRIDGE_DIR


# ==========================================================
# REGISTER CUSTOM STREAMLIT COMPONENT
# ==========================================================

component_directory = create_bridge()

xray_component = components.declare_component(
    "xray_frontend",
    path=component_directory
)


# ==========================================================
# PROCESS FRONTEND REQUESTS
# ==========================================================

if "bridge_response" not in st.session_state:

    st.session_state.bridge_response = None


if "processed_requests" not in st.session_state:

    st.session_state.processed_requests = set()


request = xray_component(
    response=st.session_state.bridge_response,
    key="xray_frontend"
)


if isinstance(request, dict):

    request_id = request.get("id")

    if (
        request_id
        and request_id not in st.session_state.processed_requests
    ):

        st.session_state.processed_requests.add(
            request_id
        )

        try:

            # Decode uploaded image

            image_data = base64.b64decode(
                request["image"]
            )

            # Check image format

            Image.open(
                io.BytesIO(image_data)
            ).verify()

            # Run prediction

            result = predict_image(image_data)

            response_data = {
                "id": request_id,
                "data": result
            }

        except UnidentifiedImageError:

            response_data = {
                "id": request_id,
                "data": {
                    "error": "Invalid image file."
                }
            }

        except Exception as e:

            response_data = {
                "id": request_id,
                "data": {
                    "error": str(e)
                }
            }


        # Send result back to original JavaScript

        st.session_state.bridge_response = response_data

        st.rerun()
