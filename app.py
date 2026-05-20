import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
IMG_SIZE    = (224, 224)
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_vgg16.h5")

# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    conv_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    conv_base.trainable = False

    model = Sequential([
        conv_base,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.load_weights(WEIGHTS_PATH)
    return model


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── Load model ───────────────────────────────────────────────────────────────
model = load_model()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI scan and the model will classify it into one of **4 categories**.")

col1, col2, col3, col4 = st.columns(4)
for col, name, color in zip(
    [col1, col2, col3, col4],
    CLASS_NAMES,
    ["#FF6B6B", "#FFA07A", "#90EE90", "#87CEEB"],
):
    col.markdown(
        f"<div style='background:{color};padding:6px 4px;border-radius:6px;"
        f"text-align:center;font-size:0.75rem;font-weight:600'>{name}</div>",
        unsafe_allow_html=True,
    )

st.divider()

uploaded = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded:
    image = Image.open(uploaded)

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    with right:
        with st.spinner("Analysing…"):
            tensor = preprocess(image)
            preds  = model.predict(tensor, verbose=0)[0]

        pred_idx   = int(np.argmax(preds))
        confidence = float(preds[pred_idx]) * 100

        st.subheader("Prediction")
        st.markdown(
            f"<h2 style='color:#4CAF50'>{CLASS_NAMES[pred_idx]}</h2>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.1f}%")

        st.subheader("Class Probabilities")
        for cls, prob in zip(CLASS_NAMES, preds):
            pct       = float(prob) * 100
            bar_color = "#4CAF50" if cls == CLASS_NAMES[pred_idx] else "#90CAF9"
            st.markdown(
                f"""
                <div style='margin-bottom:6px'>
                  <div style='display:flex;justify-content:space-between;font-size:0.85rem'>
                    <span>{cls}</span><span>{pct:.1f}%</span>
                  </div>
                  <div style='background:#e0e0e0;border-radius:4px;height:10px'>
                    <div style='width:{pct}%;background:{bar_color};
                                height:10px;border-radius:4px'></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "Model: VGG16 (ImageNet base) + custom head trained on the "
    "[Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset."
)
