import os

# ==========================================================
# ENVIRONMENT CONFIGURATION
# ==========================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import io
import base64
import traceback

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from matplotlib import colormaps


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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pneumonia_densenet121.keras"
)

BRIDGE_DIR = os.path.join(
    BASE_DIR,
    ".xray_bridge"
)


# ==========================================================
# CHECK REQUIRED FILES
# ==========================================================

required_files = [
    HTML_PATH,
    CSS_PATH,
    JS_PATH,
    MODEL_PATH
]

for path in required_files:

    if not os.path.exists(path):

        st.error(f"Required file not found: {path}")

        st.stop()


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

@st.cache_resource
def load_model():

    print("\n==================================", flush=True)

    print("LOADING DENSENET121 MODEL...", flush=True)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("MODEL LOADED SUCCESSFULLY!", flush=True)

    print("MODEL INPUT:", model.input_shape, flush=True)

    print("MODEL OUTPUT:", model.output_shape, flush=True)

    print("==================================\n", flush=True)

    return model


try:

    model = load_model()

except Exception as e:

    print("MODEL LOADING FAILED!", flush=True)

    traceback.print_exc()

    st.error(f"Model loading failed: {str(e)}")

    st.stop()


# ==========================================================
# BUILD GRAD-CAM MODEL
# ==========================================================

def build_gradcam_model(model):

    print("BUILDING GRAD-CAM MODEL...", flush=True)

    # ------------------------------------------------------
    # 1. GET DENSENET121
    # ------------------------------------------------------

    base_model = model.get_layer("densenet121")

    print("FEATURE EXTRACTOR:", base_model.name, flush=True)

    # ------------------------------------------------------
    # 2. GET LAST CONVOLUTIONAL LAYER
    # ------------------------------------------------------

    last_conv_layer = base_model.get_layer(
        "conv5_block16_concat"
    )

    print(
        "LAST CONVOLUTIONAL LAYER:",
        last_conv_layer.name,
        flush=True
    )

    # ------------------------------------------------------
    # 3. CREATE FEATURE EXTRACTOR
    # ------------------------------------------------------

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    # ------------------------------------------------------
    # 4. GET ORIGINAL MODEL LAYERS
    # ------------------------------------------------------

    rescaling_layer = next(
        layer for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Rescaling
        )
    )

    gap_layer = next(
        layer for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.GlobalAveragePooling2D
        )
    )

    dropout_layer = next(
        layer for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dropout
        )
    )

    classifier_layer = model.layers[-1]

    # ------------------------------------------------------
    # 5. REBUILD THE MODEL CONNECTION
    # ------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    # Apply original preprocessing
    x = rescaling_layer(inputs)

    # Extract convolutional features
    conv_outputs, base_outputs = feature_extractor(
        x,
        training=False
    )

    # Rebuild classification head
    x = gap_layer(base_outputs)

    x = dropout_layer(
        x,
        training=False
    )

    predictions = classifier_layer(x)

    # ------------------------------------------------------
    # 6. CREATE FINAL GRAD-CAM MODEL
    # ------------------------------------------------------

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            conv_outputs,
            predictions
        ]
    )

    print(
        "GRAD-CAM MODEL CREATED SUCCESSFULLY!",
        flush=True
    )

    print(
        "GRAD-CAM FEATURE MAP:",
        conv_outputs.shape,
        flush=True
    )

    return grad_model


try:

    gradcam_model = build_gradcam_model(model)

except Exception as e:

    gradcam_model = None

    print("GRAD-CAM MODEL CREATION FAILED!", flush=True)

    traceback.print_exc()


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image_bytes):

    print("IMAGE PREPROCESSING STARTED", flush=True)

    original_image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    resized_image = original_image.resize(
        (224, 224)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    print(
        "IMAGE SHAPE:",
        image_array.shape,
        flush=True
    )

    print("PREPROCESSING COMPLETED", flush=True)

    return original_image, image_array

# ==========================================================
# GENERATE GRAD-CAM HEATMAP
# ==========================================================

def generate_gradcam(image_array, original_image):

    if gradcam_model is None:

        raise RuntimeError(
            "Grad-CAM model could not be constructed. "
            "Check the logs for the feature extractor error."
        )

    print("GENERATING GRAD-CAM...", flush=True)

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    # ------------------------------------------------------
    # GRADIENT CALCULATION
    # ------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = gradcam_model(
            image_tensor,
            training=False
        )

        # Sigmoid output: index 0 is the pneumonia score.

        pneumonia_score = predictions[:, 0]

        loss = pneumonia_score

    # Calculate gradients of pneumonia score
    # with respect to convolutional feature maps.

    gradients = tape.gradient(
        loss,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Could not calculate Grad-CAM gradients."
        )

    # ------------------------------------------------------
    # GLOBAL AVERAGE POOLING OF GRADIENTS
    # ------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # ------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # ------------------------------------------------------

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # ReLU removes negative influence.

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize heatmap.

    maximum = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        maximum + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    print("GRAD-CAM HEATMAP CREATED", flush=True)

    return heatmap


# ==========================================================
# CREATE HEATMAP OVERLAY
# ==========================================================

def create_gradcam_overlay(original_image, heatmap):

    print("CREATING GRAD-CAM OVERLAY...", flush=True)

    original_image = original_image.convert("RGB")

    # Resize heatmap to original image dimensions.

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap_image,
        dtype=np.uint8
    )

    # Apply JET colormap.

    colored_heatmap = colormaps["jet"](
        heatmap_array / 255.0
    )

    colored_heatmap = np.uint8(
        colored_heatmap[:, :, :3] * 255
    )

    colored_image = Image.fromarray(
        colored_heatmap
    )

    # Blend original X-ray and heatmap.

    overlay = Image.blend(
        original_image,
        colored_image,
        alpha=0.40
    )

    # Keep generated image reasonably sized.

    overlay.thumbnail(
        (1200, 1200)
    )

    # Convert to PNG bytes.

    buffer = io.BytesIO()

    overlay.save(
        buffer,
        format="PNG"
    )

    image_bytes = buffer.getvalue()

    # Convert image into Base64.

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    print("GRAD-CAM OVERLAY CREATED SUCCESSFULLY", flush=True)

    return encoded_image


# ==========================================================
# PREDICTION + GRAD-CAM
# ==========================================================

def predict_image(image_bytes):

    print("\n==================================", flush=True)

    print("NEW X-RAY IMAGE RECEIVED", flush=True)

    original_image, image_array = preprocess_image(
        image_bytes
    )

    # ------------------------------------------------------
    # MODEL PREDICTION
    # ------------------------------------------------------

    if gradcam_model is not None:

        try:

            print("PREDICTION AND GRAD-CAM STARTED", flush=True)

            image_tensor = tf.convert_to_tensor(
                image_array,
                dtype=tf.float32
            )

            with tf.GradientTape() as tape:

                conv_outputs, predictions = gradcam_model(
                    image_tensor,
                    training=False
                )

                pneumonia_score_tensor = predictions[:, 0]

            gradients = tape.gradient(
                pneumonia_score_tensor,
                conv_outputs
            )

            if gradients is None:

                raise ValueError(
                    "Gradients could not be calculated."
                )

            # Grad-CAM calculation

            pooled_gradients = tf.reduce_mean(
                gradients,
                axis=(0, 1, 2)
            )

            conv_features = conv_outputs[0]

            heatmap = tf.reduce_sum(
                conv_features * pooled_gradients,
                axis=-1
            )

            heatmap = tf.maximum(
                heatmap,
                0
            )

            heatmap = heatmap / (
                tf.reduce_max(heatmap) + 1e-8
            )

            heatmap = heatmap.numpy()

            pneumonia_score = float(
                predictions.numpy().flatten()[0]
            )

            gradcam_image = create_gradcam_overlay(
                original_image,
                heatmap
            )

            gradcam_message = (
                "Grad-CAM visualization generated successfully."
            )

        except Exception as e:

            print("GRAD-CAM GENERATION FAILED!", flush=True)

            traceback.print_exc()

            # Fall back to ordinary prediction.

            predictions = model.predict(
                image_array,
                verbose=0
            )

            pneumonia_score = float(
                np.asarray(predictions).flatten()[0]
            )

            gradcam_image = None

            gradcam_message = (
                "Prediction completed, but Grad-CAM failed: "
                + str(e)
            )

    else:

        predictions = model.predict(
            image_array,
            verbose=0
        )

        pneumonia_score = float(
            np.asarray(predictions).flatten()[0]
        )

        gradcam_image = None

        gradcam_message = (
            "Grad-CAM could not be initialized."
        )

    # ------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------

    pneumonia_score = float(
        np.clip(pneumonia_score, 0, 1)
    )

    normal_score = 1.0 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "PNEUMONIA"

        description = (
            "The model detected image patterns associated "
            "with pneumonia."
        )

    else:

        result = "NORMAL"

        description = (
            "The model classified this X-ray image as normal."
        )

    # ------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------

    response = {

        "prediction": result,

        "normal_score": round(
            normal_score * 100, 2
        ),

        "pneumonia_score": round(
            pneumonia_score * 100, 2
        ),

        "description": description,

        "gradcam": gradcam_image,

        "gradcam_message": gradcam_message

    }

    print("PREDICTION COMPLETED!", flush=True)

    print("RESULT:", result, flush=True)

    print(
        "PNEUMONIA SCORE:",
        response["pneumonia_score"],
        flush=True
    )

    print(
        "GRAD-CAM AVAILABLE:",
        gradcam_image is not None,
        flush=True
    )

    print("==================================\n", flush=True)

    return response


# ==========================================================
# CREATE FRONTEND BRIDGE
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
    # STREAMLIT JAVASCRIPT BRIDGE
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


    // Component ready

    sendToStreamlit(
        "streamlit:componentReady",
        {
            apiVersion: 1
        }
    );


    const pendingRequests = new Map();


    // Receive Python response

    window.addEventListener("message", function(event) {

        const message = event.data;

        if (!message) {
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

                console.log(
                    "PREDICTION RESPONSE RECEIVED",
                    response.data
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

            return new Promise((resolve, reject) => {

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
                            resolve: resolve,
                            reject: reject
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

                    reject(
                        new Error("Unable to read uploaded image.")
                    );

                };


                reader.readAsDataURL(file);

            });

        }

        return originalFetch(input, init);

    };


    // ------------------------------------------------------
    // COMPONENT HEIGHT
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
    # INSERT JAVASCRIPT
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

    component_html = os.path.join(
        BRIDGE_DIR,
        "index.html"
    )

    with open(
        component_html,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(html)

    print("FRONTEND BRIDGE CREATED", flush=True)

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
# RECEIVE IMAGE FROM FRONTEND
# ==========================================================

request = xray_component(
    response=st.session_state.bridge_response,
    key="xray_frontend"
)


# ==========================================================
# PROCESS IMAGE REQUEST
# ==========================================================

if isinstance(request, dict):

    request_id = request.get("id")

    if (
        request_id
        and request_id not in st.session_state.processed_requests
    ):

        st.session_state.processed_requests.add(
            request_id
        )

        print("NEW IMAGE REQUEST RECEIVED", flush=True)

        try:

            # Decode image

            image_data = base64.b64decode(
                request["image"],
                validate=True
            )

            # File-size validation

            if len(image_data) > 10 * 1024 * 1024:

                raise ValueError(
                    "Image must be smaller than 10 MB."
                )

            # Validate image

            image = Image.open(
                io.BytesIO(image_data)
            )

            image.verify()

            # Run prediction and Grad-CAM

            result = predict_image(
                image_data
            )

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

            print("REQUEST FAILED!", flush=True)

            traceback.print_exc()

            response_data = {

                "id": request_id,

                "data": {

                    "error": str(e)

                }

            }

        # Send response to frontend

        st.session_state.bridge_response = response_data

        print(
            "SENDING PREDICTION AND GRAD-CAM TO FRONTEND",
            flush=True
        )

        st.rerun()
