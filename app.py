import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import os


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------
# LOAD MODEL
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "pneumonia_densenet121.keras"
)


@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


try:

    model = load_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# --------------------------------
# IMAGE PREPROCESSING
# --------------------------------

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
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


# --------------------------------
# PREDICTION FUNCTION
# --------------------------------

def predict_image(image):

    image_array = preprocess_image(image)

    prediction = model.predict(
        image_array,
        verbose=0
    )

    prediction = np.asarray(prediction)

    if prediction.size == 1:

        pneumonia_score = float(
            prediction.reshape(-1)[0]
        )

    elif prediction.shape[-1] == 2:

        pneumonia_score = float(
            prediction.reshape(-1, 2)[0][1]
        )

    else:

        raise ValueError("Unexpected model output shape")

    pneumonia_score = np.clip(
        pneumonia_score,
        0,
        1
    )

    normal_score = 1 - pneumonia_score

    if pneumonia_score >= 0.5:

        result = "PNEUMONIA"

    else:

        result = "NORMAL"

    return result, normal_score, pneumonia_score


# --------------------------------
# APPLICATION INTERFACE
# --------------------------------

st.title("🩺 Chest X-ray Pneumonia Detection")

st.write(
    "An AI-powered chest X-ray image classification "
    "system using DenseNet121."
)

st.info(
    "Upload a chest X-ray image to get a model prediction."
)


# --------------------------------
# IMAGE UPLOAD
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded X-ray")

        st.image(
            image,
            caption="Your uploaded chest X-ray",
            use_container_width=True
        )

    with col2:

        st.subheader("AI Analysis")

        if st.button(
            "Analyze X-ray",
            use_container_width=True
        ):

            with st.spinner("Analyzing X-ray..."):

                try:

                    result, normal, pneumonia = predict_image(
                        image
                    )

                    if result == "PNEUMONIA":

                        st.error("Prediction: PNEUMONIA")

                    else:

                        st.success("Prediction: NORMAL")

                    st.metric(
                        "Normal Score",
                        f"{normal * 100:.2f}%"
                    )

                    st.metric(
                        "Pneumonia Score",
                        f"{pneumonia * 100:.2f}%"
                    )

                    st.subheader("Prediction Confidence")

                    st.progress(float(pneumonia))

                    st.write(
                        "Pneumonia score:",
                        f"{pneumonia * 100:.2f}%"
                    )

                    st.progress(float(normal))

                    st.write(
                        "Normal score:",
                        f"{normal * 100:.2f}%"
                    )

                except Exception as e:

                    st.error(f"Prediction failed: {e}")


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Developed using Python, TensorFlow and DenseNet121."
)

st.warning(
    "Educational and research project only. "
    "This AI prediction is not a medical diagnosis."
)
