from pathlib import Path
import base64
import shutil
import uuid
import io

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "pneumonia_densenet121.keras"
HTML_PATH = ROOT / "index.html"
STATIC_DIR = ROOT / "static"
COMPONENT_FRONTEND = ROOT / "clarityx_component" / "frontend"


def prepare_component_frontend():
    """Build a browser-ready copy of the user's original frontend.

    The original index.html, CSS and JS in the project root are not edited.
    Only the generated copy inside clarityx_component/frontend gets local
    asset paths and the small Streamlit bridge script.
    """
    COMPONENT_FRONTEND.mkdir(parents=True, exist_ok=True)
    html = HTML_PATH.read_text(encoding="utf-8")

    html = html.replace(
        "{{ url_for('static', filename='style.css') }}",
        "style.css",
    )
    original_script_tag = '<script src="{{ url_for(\'static\', filename=\'script.js\', v=2) }}"></script>'
    replacement_script_tag = (
        '<script src="bridge.js"></script>\n'
        '<script src="script.js"></script>'
    )
    if original_script_tag not in html:
        raise RuntimeError(
            "The expected script.js reference was not found in index.html. "
            "Please keep the original frontend files and check their names."
        )
    html = html.replace(original_script_tag, replacement_script_tag)

    (COMPONENT_FRONTEND / "index.html").write_text(html, encoding="utf-8")
    shutil.copy2(STATIC_DIR / "style.css", COMPONENT_FRONTEND / "style.css")
    shutil.copy2(STATIC_DIR / "script.js", COMPONENT_FRONTEND / "script.js")

    bridge = r'''(() => {
  const pending = new Map();
  let latestArgs = {};

  function sendToStreamlit(value) {
    window.parent.postMessage({
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value,
      dataType: "json"
    }, "*");
  }

  function setFrameHeight() {
    const height = Math.max(
      document.documentElement.scrollHeight,
      document.body ? document.body.scrollHeight : 0
    );
    window.parent.postMessage({
      isStreamlitMessage: true,
      type: "streamlit:setFrameHeight",
      height
    }, "*");
  }

  window.addEventListener("message", (event) => {
    const message = event.data;
    if (!message || message.type !== "streamlit:render") return;
    latestArgs = message.args || {};
    const result = latestArgs.result;
    if (result && result.request_id && pending.has(result.request_id)) {
      const resolve = pending.get(result.request_id);
      pending.delete(result.request_id);
      const payload = result.payload || {};
      resolve({
        ok: result.ok !== false,
        status: result.ok === false ? 400 : 200,
        json: async () => payload
      });
    }
    setTimeout(setFrameHeight, 80);
  });

  const nativeFetch = window.fetch.bind(window);
  window.fetch = function(input, init = {}) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (!url.endsWith("/predict") || !init.body || !(init.body instanceof FormData)) {
      return nativeFetch(input, init);
    }

    const file = init.body.get("image");
    if (!(file instanceof Blob)) {
      return Promise.resolve({ ok: false, status: 400, json: async () => ({error: "Please select an image."}) });
    }

    const requestId = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : String(Date.now()) + Math.random().toString(16).slice(2);

    return new Promise((resolve) => {
      pending.set(requestId, resolve);
      const reader = new FileReader();
      reader.onload = () => {
        sendToStreamlit({
          request_id: requestId,
          filename: file.name || "chest_xray.png",
          mime_type: file.type || "image/png",
          image_data: reader.result
        });
      };
      reader.onerror = () => {
        pending.delete(requestId);
        resolve({ok: false, status: 400, json: async () => ({error: "Could not read the selected image."})});
      };
      reader.readAsDataURL(file);
    });
  };

  window.addEventListener("load", () => {
    setFrameHeight();
    if (window.ResizeObserver) {
      new ResizeObserver(setFrameHeight).observe(document.body);
    }
  });

  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:componentReady",
    apiVersion: 1
  }, "*");
})();
'''
    (COMPONENT_FRONTEND / "bridge.js").write_text(bridge, encoding="utf-8")


@st.cache_resource(show_spinner="Loading the DenseNet121 model...")
def load_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Put pneumonia_densenet121.keras in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_image(model, request):
    request_id = request.get("request_id", str(uuid.uuid4()))
    try:
        data_url = request.get("image_data", "")
        if "," not in data_url:
            raise ValueError("The uploaded image could not be read.")
        encoded = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded)

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB").resize((224, 224))
            image_array = np.asarray(image, dtype=np.float32)

        # Do not divide by 255 here: the model itself contains the Rescaling layer.
        image_array = np.expand_dims(image_array, axis=0)
        raw_score = float(model.predict(image_array, verbose=0)[0][0])
        pneumonia_score = float(np.clip(raw_score * 100.0, 0.0, 100.0))
        normal_score = 100.0 - pneumonia_score

        if raw_score >= 0.5:
            prediction = "PNEUMONIA"
            description = (
                "The model output is higher for the Pneumonia class. "
                "This is an experimental research result, not a diagnosis."
            )
        else:
            prediction = "NORMAL"
            description = (
                "The model output is higher for the Normal class. "
                "This is an experimental research result, not a diagnosis."
            )

        payload = {
            "prediction": prediction,
            "normal_score": round(normal_score, 2),
            "pneumonia_score": round(pneumonia_score, 2),
            "description": description,
            "gradcam": None,
            "gradcam_message": "Grad-CAM is turned off in this initial Streamlit deployment to reduce memory use."
        }
        return {"request_id": request_id, "ok": True, "payload": payload}

    except Exception as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "payload": {"error": f"Prediction error: {exc}"}
        }


st.set_page_config(page_title="ClarityX | AI Chest X-Ray Analysis", layout="wide")

try:
    prepare_component_frontend()
    from clarityx_component import clarityx_component
except Exception as exc:
    st.error(f"Could not prepare the ClarityX website: {exc}")
    st.stop()

if "clarityx_result" not in st.session_state:
    st.session_state.clarityx_result = None
if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None

request = clarityx_component(
    result=st.session_state.clarityx_result,
    key="clarityx_frontend",
    default=None,
)

if isinstance(request, dict):
    request_id = request.get("request_id")
    if request_id and request_id != st.session_state.last_request_id:
        try:
            model = load_model()
            st.session_state.clarityx_result = predict_image(model, request)
        except Exception as exc:
            st.session_state.clarityx_result = {
                "request_id": request_id,
                "ok": False,
                "payload": {"error": str(exc)}
            }
        st.session_state.last_request_id = request_id
        st.rerun()
