import os

# ==========================================================
# ENVIRONMENT CONFIGURATION
# ==========================================================

# Force CPU execution on Streamlit Cloud
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import base64
import io
import uuid
import traceback

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
# CHECK FILES
# ==========================================================

for path in [HTML_PATH, CSS_PATH, JS_PATH]:

    if not os.path.exists(path):
        st.error(f"Required frontend file not found: {path}")
        st.stop()


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    print("====================================", flush=True)
    print("LOADING DENSENET121 MODEL...", flush=True)

    model_path = next(
        (path for path in MODEL_PATHS if os.path.exists(path)),
        None
    )

    if model_path is None:

        raise FileNotFoundError(
            "Model file not found. Expected "
            "pneumonia_densenet121.keras in the root "
            "or models folder."
        )

    print(f"MODEL PATH: {model_path}", flush=True)

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    print("MODEL LOADED SUCCESSFULLY!", flush=True)
    print(f"MODEL INPUT: {model.input_shape}", flush=True)
    print(f"MODEL OUTPUT: {model.output_shape}", flush=True)
    print("====================================", flush=True)

    return model


try:

    model = load_model()

except Exception as e:

    print("MODEL LOADING FAILED!", flush=True)
    traceback.print_exc()

    st.error(f"Model loading failed: {str(e)}")
    st.stop()


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_bytes):

    print("IMAGE PREPROCESSING STARTED", flush=True)

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    print(f"ORIGINAL IMAGE SIZE: {image.size}", flush=True)

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    print(
        f"PREPROCESSED IMAGE SHAPE: {image_array.shape}",
        flush=True
    )

    print("IMAGE PREPROCESSING COMPLETED", flush=True)

    return image_array


# IMPORTANT:
# This assumes your model was trained using:
# RGB images, 224x224 dimensions, pixel values 0-255.
# Your actual training preprocessing must match this.


# ==========================================================
# MODEL PREDICTION
# ==========================================================

def predict_image(image_bytes):

    print("====================================", flush=True)
    print("IMAGE RECEIVED BY MODEL BACKEND", flush=True)

    image_array = preprocess_image(image_bytes)

    print("PREDICTION STARTED", flush=True)

    with tf.device("/CPU:0"):

        prediction = model.predict(
            image_array,
            verbose=0
        )

    print("RAW MODEL OUTPUT:", prediction, flush=True)

    prediction = np.asarray(prediction).flatten()

    if len(prediction) == 1:

        pneumonia_score = float(prediction[0])

    elif len(prediction) == 2:

        pneumonia_score = float(prediction[1])

    else:

        raise ValueError(
            f"Unexpected model output: {prediction}"
        )

    pneumonia_score = float(
        np.clip(pneumonia_score, 0, 1)
    )

    normal_score = 1.0 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "Pneumonia Detected"

        description = (
            "The model detected patterns associated "
            "with pneumonia in this X-ray image."
        )

    else:

        result = "Normal"

        description = (
            "The model classified this X-ray image "
            "as normal."
        )

    response = {

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
            "in this backend."
        )
    }

    print("PREDICTION COMPLETED SUCCESSFULLY!", flush=True)
    print("RESULT:", result, flush=True)
    print("NORMAL SCORE:", response["normal_score"], flush=True)
    print(
        "PNEUMONIA SCORE:",
        response["pneumonia_score"],
        flush=True
    )
    print("====================================", flush=True)

    return response


# ==========================================================
# CREATE FRONTEND BRIDGE
# ==========================================================

def create_bridge():

    os.makedirs(
        BRIDGE_DIR,
        exist_ok=True
    )

    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css = f.read()

    with open(JS_PATH, "r", encoding="utf-8") as f:
        javascript = f.read()

    # ------------------------------------------------------
    # STREAMLIT COMMUNICATION BRIDGE
    # ------------------------------------------------------

    bridge_javascript = r"""
    <script>

    function sendToStreamlit(type, data = {}) {

        window.parent.postMessage({

            isStreamlitMessage: true,
            type: type,
            ...data

        }, "*");

    }


    // Tell Streamlit that the component is ready

    sendToStreamlit(
        "streamlit:componentReady",
        {
            apiVersion: 1
        }
    );


    // Store pending prediction requests

    const pendingRequests = new Map();


    // Receive Python responses

    window.addEventListener("message", function(event) {

        const message = event.data;

        if (!message || !message.isStreamlitMessage) {
            return;
        }

        if (message.type === "streamlit:render") {

            const response = message.args?.response;

            if (!response || !response.id) {
                return;
            }

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
                            status: response.data.error ? 500 : 200,
                            headers: {
                                "Content-Type": "application/json"
                            }
                        }
                    )
                );

            }

        }

    });


    // ------------------------------------------------------
    // INTERCEPT ORIGINAL FETCH
    // ------------------------------------------------------

    const originalFetch = window.fetch.bind(window);

    window.fetch = async function(input, init = {}) {

        const url = typeof input === "string"
            ? input
            : input.url;

        if (
            url.endsWith("/predict") &&
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
                        crypto.randomUUID
                            ? crypto.randomUUID()
                            : Date.now().toString() +
                              Math.random().toString(36).slice(2);


                    pendingRequests.set(
                        requestId,
                        {
                            resolve: resolve
                        }
                    );


                    sendToStreamlit(
                        "streamlit:setComponentValue",
                        {
                            value: {

                                id: requestId,

                                filename: file.name,

                                image: base64Image

                            }
                        }
                    );

                };


                reader.onerror = function() {

                    resolve(
                        new Response(
                            JSON.stringify({
                                error: "Unable to read image."
                            }),
                            {
                                status: 500,
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


    // ------------------------------------------------------
    // UPDATE COMPONENT HEIGHT
    // ------------------------------------------------------

    function updateHeight() {

        const height = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight
        );

        sendToStreamlit(
            "streamlit:setFrameHeight",
            {
                height: height
            }
        );

    }


    window.addEventListener("load", updateHeight);
    window.addEventListener("resize", updateHeight);

    new ResizeObserver(
        updateHeight
    ).observe(document.body);

    </script>
    """

    # ------------------------------------------------------
    # INSERT CSS
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
    # INSERT BRIDGE BEFORE ORIGINAL JAVASCRIPT
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
    # SAVE GENERATED HTML
    # ------------------------------------------------------

    with open(
        os.path.join(BRIDGE_DIR, "index.html"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html)

    print("FRONTEND BRIDGE CREATED SUCCESSFULLY", flush=True)

    return BRIDGE_DIR


# ==========================================================
# REGISTER STREAMLIT COMPONENT
# ==========================================================

component_directory = create_bridge()

xray_component = components.declare_component(
    "xray_frontend",
    path=component_directory
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "bridge_response" not in st.session_state:

    st.session_state.bridge_response = None


if "processed_requests" not in st.session_state:

    st.session_state.processed_requests = set()


# ==========================================================
# RECEIVE FRONTEND REQUEST
# ==========================================================

request = xray_component(
    response=st.session_state.bridge_response,
    key="xray_frontend"
)


# ==========================================================
# PROCESS UPLOADED IMAGE
# ==========================================================

if isinstance(request, dict):

    request_id = request.get("id")

    if (
        request_id
        and request_id not in st.session_state.processed_requests
    ):

        st.session_state.processed_requests.add(request_id)

        print("NEW REQUEST RECEIVED", flush=True)

        try:

            # ----------------------------------------------
            # CHECK IMAGE SIZE
            # ----------------------------------------------

            image_data = base64.b64decode(
                request["image"],
                validate=True
            )

            if len(image_data) > 10 * 1024 * 1024:

                raise ValueError(
                    "Image size exceeds the 10 MB limit."
                )

            # ----------------------------------------------
            # VALIDATE IMAGE
            # ----------------------------------------------

            image = Image.open(
                io.BytesIO(image_data)
            )

            image.verify()

            print(
                "IMAGE VALIDATION SUCCESSFUL",
                flush=True
            )

            # ----------------------------------------------
            # RUN MODEL
            # ----------------------------------------------

            result = predict_image(image_data)

            response_data = {

                "id": request_id,

                "data": result

            }

        except Exception as e:

            print("PREDICTION ERROR!", flush=True)

            traceback.print_exc()

            response_data = {

                "id": request_id,

                "data": {

                    "error": str(e)

                }

            }

        # ----------------------------------------------
        # RETURN RESPONSE TO JAVASCRIPT
        # ----------------------------------------------

        st.session_state.bridge_response = response_data

        print(
            "SENDING RESPONSE TO FRONTEND",
            flush=True
        )

        st.rerun()
