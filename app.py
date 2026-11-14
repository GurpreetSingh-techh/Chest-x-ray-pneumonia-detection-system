import os
import io
import re
import base64
import uuid
import shutil
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import numpy as np
from PIL import Image


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ClarityX | AI Chest X-Ray Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ROOT = Path(__file__).parent

MODEL_PATH = ROOT / "pneumonia_densenet121.keras"

# Automatically find index.html
if (ROOT / "templates" / "index.html").exists():
    HTML_PATH = ROOT / "templates" / "index.html"
else:
    HTML_PATH = ROOT / "index.html"

STATIC_DIR = ROOT / "static"

COMPONENT_DIR = ROOT / "clarityx_component"

FRONTEND_DIR = COMPONENT_DIR / "frontend"

IMAGE_SIZE = (224, 224)


# ============================================================
# 2. CREATE CUSTOM STREAMLIT COMPONENT FRONTEND
# ============================================================

def prepare_frontend():

    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

    if not HTML_PATH.exists():
        raise FileNotFoundError("index.html was not found.")

    if not (STATIC_DIR / "style.css").exists():
        raise FileNotFoundError("static/style.css was not found.")

    if not (STATIC_DIR / "script.js").exists():
        raise FileNotFoundError("static/script.js was not found.")

    html = HTML_PATH.read_text(encoding="utf-8")

    # Replace Flask CSS reference with local CSS file
    html = re.sub(
        r"\{\{\s*url_for\(['\"]static['\"],\s*filename=['\"]style\.css['\"]\)\s*\}\}",
        "style.css",
        html
    )

    # Replace Flask JavaScript reference
    script_pattern = (
        r'<script\s+src=["\']'
        r'\{\{\s*url_for\([\'"]static[\'"],\s*filename=[\'"]script\.js[\'"]'
        r'(?:,\s*v=\d+)?\)\s*\}\}'
        r'["\']\s*>\s*</script>'
    )

    replacement = """
    <script src="bridge.js"></script>
    <script src="script.js"></script>
    """

    html, count = re.subn(
        script_pattern,
        replacement,
        html,
        count=1
    )

    if count == 0:
        raise RuntimeError(
            "Could not find the original script.js reference in index.html."
        )

    # Save a browser-ready copy
    (FRONTEND_DIR / "index.html").write_text(
        html,
        encoding="utf-8"
    )

    shutil.copy2(
        STATIC_DIR / "style.css",
        FRONTEND_DIR / "style.css"
    )

    shutil.copy2(
        STATIC_DIR / "script.js",
        FRONTEND_DIR / "script.js"
    )

    # --------------------------------------------------------
    # JAVASCRIPT - STREAMLIT COMMUNICATION BRIDGE
    # --------------------------------------------------------

    bridge_js = r"""
(() => {

    const pendingRequests = new Map();

    // Send uploaded image information to Python
    function sendToPython(value) {

        window.parent.postMessage({

            isStreamlitMessage: true,

            type: "streamlit:setComponentValue",

            value: value,

            dataType: "json"

        }, "*");

    }


    // Calculate complete webpage height
    function setFrameHeight() {

        const body = document.body;

        const html = document.documentElement;

        const height = Math.ceil(Math.max(

            body.scrollHeight,

            body.offsetHeight,

            body.clientHeight,

            html.scrollHeight,

            html.offsetHeight,

            html.clientHeight,

            body.getBoundingClientRect().height,

            html.getBoundingClientRect().height

        )) + 50;


        window.parent.postMessage({

            isStreamlitMessage: true,

            type: "streamlit:setFrameHeight",

            height: height

        }, "*");

    }


    // Receive prediction result from Python
    window.addEventListener("message", (event) => {

        const message = event.data;

        if (!message || message.type !== "streamlit:render") {
            return;
        }

        const args = message.args || {};

        const result = args.result;

        if (
            result &&
            result.request_id &&
            pendingRequests.has(result.request_id)
        ) {

            const resolve = pendingRequests.get(
                result.request_id
            );

            pendingRequests.delete(result.request_id);

            resolve({

                ok: result.ok !== false,

                status: result.ok === false ? 400 : 200,

                json: async () => result.payload || {}

            });

        }

        setTimeout(setFrameHeight, 100);

    });


    // Keep original JavaScript fetch functionality
    const originalFetch = window.fetch.bind(window);


    window.fetch = function(input, init = {}) {

        const url = typeof input === "string"
            ? input
            : (input && input.url) || "";


        // Intercept only the original prediction request
        if (
            !url.endsWith("/predict") ||
            !init.body ||
            !(init.body instanceof FormData)
        ) {

            return originalFetch(input, init);

        }


        const file = init.body.get("image");


        if (!(file instanceof Blob)) {

            return Promise.resolve({

                ok: false,

                status: 400,

                json: async () => ({
                    error: "Please select an image."
                })

            });

        }


        const requestId = (
            window.crypto && crypto.randomUUID
        )
            ? crypto.randomUUID()
            : String(Date.now()) +
              Math.random().toString(16).slice(2);


        return new Promise((resolve) => {

            pendingRequests.set(requestId, resolve);

            const reader = new FileReader();


            reader.onload = () => {

                sendToPython({

                    request_id: requestId,

                    filename: file.name || "xray.png",

                    mime_type: file.type || "image/png",

                    image_data: reader.result

                });

            };


            reader.onerror = () => {

                pendingRequests.delete(requestId);

                resolve({

                    ok: false,

                    status: 400,

                    json: async () => ({
                        error: "Could not read the image."
                    })

                });

            };


            reader.readAsDataURL(file);

        });

    };


    // --------------------------------------------------------
    // DYNAMIC IFRAME HEIGHT
    // --------------------------------------------------------

    function updateHeightRepeatedly() {

        setFrameHeight();

        [100, 300, 700, 1500, 2500, 4000].forEach(delay => {

            setTimeout(setFrameHeight, delay);

        });

    }


    window.addEventListener("load", () => {

        updateHeightRepeatedly();


        // Watch changes in webpage size
        if (window.ResizeObserver) {

            const resizeObserver = new ResizeObserver(() => {

                setFrameHeight();

            });

            resizeObserver.observe(document.body);

            resizeObserver.observe(
                document.documentElement
            );

        }


        // Watch dynamically added content
        if (window.MutationObserver) {

            const mutationObserver = new MutationObserver(() => {

                setFrameHeight();

            });

            mutationObserver.observe(document.body, {

                childList: true,

                subtree: true,

                attributes: true

            });

        }


        // Recalculate after images load
        document.querySelectorAll("img").forEach(img => {

            img.addEventListener("load", setFrameHeight);

        });


        // Recalculate after browser resizing
        window.addEventListener("resize", setFrameHeight);

    });


    // Tell Streamlit that component is ready
    window.parent.postMessage({

        isStreamlitMessage: true,

        type: "streamlit:componentReady",

        apiVersion: 1

    }, "*");

})();
"""

    (FRONTEND_DIR / "bridge.js").write_text(
        bridge_js,
        encoding="utf-8"
    )


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():

    import tensorflow as tf

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "pneumonia_densenet121.keras was not found."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("MODEL LOADED SUCCESSFULLY")

    print("MODEL INPUT:", model.input_shape)

    print("MODEL OUTPUT:", model.output_shape)

    return model


# ============================================================
# 4. FIND DENSENET121 BACKBONE AND LAST CONVOLUTION LAYER
# ============================================================

def find_backbone(model):

    import tensorflow as tf

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if "densenet" in layer.name.lower():

                return layer

            for nested_layer in layer.layers:

                if isinstance(nested_layer, tf.keras.layers.Conv2D):

                    return layer

    raise ValueError(
        "DenseNet121 backbone was not found in the model."
    )


def find_last_conv_layer(backbone):

    import tensorflow as tf

    conv_layers = []

    for layer in backbone.layers:

        if isinstance(layer, tf.keras.layers.Conv2D):

            conv_layers.append(layer)

        elif isinstance(layer, tf.keras.Model):

            for nested_layer in layer.layers:

                if isinstance(
                    nested_layer,
                    tf.keras.layers.Conv2D
                ):

                    conv_layers.append(nested_layer)

    if not conv_layers:

        raise ValueError(
            "No convolution layer found in DenseNet121."
        )

    return conv_layers[-1]


# ============================================================
# 5. BUILD GRAD-CAM MODEL
# ============================================================

@st.cache_resource
def build_gradcam_model():

    import tensorflow as tf

    model = load_model()

    backbone = find_backbone(model)

    last_conv = find_last_conv_layer(backbone)

    feature_extractor = tf.keras.Model(

        inputs=backbone.input,

        outputs=[
            last_conv.output,
            backbone.output
        ]

    )

    backbone_index = model.layers.index(backbone)

    classifier_layers = model.layers[backbone_index + 1:]


    print("GRAD-CAM BACKBONE:", backbone.name)

    print("GRAD-CAM LAYER:", last_conv.name)

    print("GRAD-CAM MODEL READY")


    return (
        feature_extractor,
        classifier_layers
    )


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    import tensorflow as tf

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # DenseNet121 preprocessing
    image_array = tf.keras.applications.densenet.preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# 7. GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(image_array, prediction_class):

    import tensorflow as tf

    feature_extractor, classifier_layers = build_gradcam_model()

    input_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = feature_extractor(
            input_tensor,
            training=False
        )

        tape.watch(conv_outputs)

        x = backbone_output

        for layer in classifier_layers:

            x = layer(x, training=False)

        predictions = x

        if prediction_class == "PNEUMONIA":

            loss = predictions[:, 0]

        else:

            loss = 1.0 - predictions[:, 0]


    gradients = tape.gradient(
        loss,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated for Grad-CAM."
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

    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# ============================================================
# 8. CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam_image(original_image, heatmap):

    import matplotlib.cm as cm

    original_image = original_image.convert("RGB")

    original_image = original_image.resize(IMAGE_SIZE)

    original_array = np.asarray(
        original_image,
        dtype=np.float32
    )

    heatmap_image = Image.fromarray(

        np.uint8(heatmap * 255)

    )

    heatmap_image = heatmap_image.resize(
        IMAGE_SIZE
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0


    # JET colour map
    jet = cm.get_cmap("jet")

    coloured_heatmap = jet(
        heatmap_array
    )[:, :, :3]

    coloured_heatmap = coloured_heatmap * 255


    # Overlay heatmap on original X-ray
    overlay = (
        0.60 * original_array +
        0.40 * coloured_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)


    result_image = Image.fromarray(overlay)

    buffer = io.BytesIO()

    result_image.save(
        buffer,
        format="PNG"
    )

    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded_image


# ============================================================
# 9. PREDICT IMAGE
# ============================================================

def predict_image(model, request):

    request_id = request.get(
        "request_id",
        str(uuid.uuid4())
    )

    try:

        data_url = request.get(
            "image_data",
            ""
        )

        if "," not in data_url:

            raise ValueError(
                "Invalid image data."
            )

        image_bytes = base64.b64decode(
            data_url.split(",", 1)[1]
        )

        original_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        # Preprocessing
        image_array = preprocess_image(
            original_image
        )


        print("IMAGE PREPROCESSING COMPLETED")


        # Prediction
        raw_prediction = model.predict(
            image_array,
            verbose=0
        )

        raw_score = float(
            raw_prediction[0][0]
        )


        pneumonia_score = raw_score * 100

        normal_score = (1 - raw_score) * 100


        if raw_score >= 0.5:

            prediction = "PNEUMONIA"

            prediction_class = "PNEUMONIA"

            description = (
                "The model output is higher for the "
                "Pneumonia category."
            )

        else:

            prediction = "NORMAL"

            prediction_class = "NORMAL"

            description = (
                "The model output is higher for the "
                "Normal category."
            )


        print("PREDICTION:", prediction)


        # ----------------------------------------------------
        # GRAD-CAM GENERATION
        # ----------------------------------------------------

        gradcam_base64 = None

        gradcam_message = None

        try:

            heatmap = generate_gradcam(
                image_array,
                prediction_class
            )

            gradcam_base64 = create_gradcam_image(
                original_image,
                heatmap
            )

            print("GRAD-CAM GENERATED SUCCESSFULLY")

        except Exception as gradcam_error:

            gradcam_message = str(
                gradcam_error
            )

            print(
                "GRAD-CAM ERROR:",
                gradcam_message
            )


        # ----------------------------------------------------
        # RESPONSE TO ORIGINAL JAVASCRIPT
        # ----------------------------------------------------

        payload = {

            "prediction": prediction,

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


        return {

            "request_id": request_id,

            "ok": True,

            "payload": payload

        }


    except Exception as error:

        print(
            "PREDICTION ERROR:",
            str(error)
        )

        return {

            "request_id": request_id,

            "ok": False,

            "payload": {

                "error": str(error)

            }

        }


# ============================================================
# 10. DECLARE STREAMLIT COMPONENT
# ============================================================

try:

    prepare_frontend()

    clarityx_component = components.declare_component(

        "clarityx_frontend",

        path=str(FRONTEND_DIR)

    )

except Exception as error:

    st.error(
        f"Frontend initialization error: {error}"
    )

    st.stop()


# ============================================================
# 11. SESSION STATE
# ============================================================

if "clarityx_result" not in st.session_state:

    st.session_state.clarityx_result = None


if "last_request_id" not in st.session_state:

    st.session_state.last_request_id = None


# ============================================================
# 12. DISPLAY COMPLETE WEBPAGE
# ============================================================

request = clarityx_component(

    result=st.session_state.clarityx_result,

    key="clarityx_frontend",

    default=None

)


# ============================================================
# 13. HANDLE IMAGE UPLOAD AND PREDICTION
# ============================================================

if isinstance(request, dict):

    request_id = request.get(
        "request_id"
    )

    if (
        request_id and
        request_id != st.session_state.last_request_id
    ):

        try:

            model = load_model()

            result = predict_image(
                model,
                request
            )

            st.session_state.clarityx_result = result

        except Exception as error:

            st.session_state.clarityx_result = {

                "request_id": request_id,

                "ok": False,

                "payload": {

                    "error": str(error)

                }

            }


        st.session_state.last_request_id = request_id

        st.rerun()
