from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from preprocessing import DATA_DIR, create_demo_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_labels(label_path: str):
    with open(label_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_image(image: Image.Image):
    img = image.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_breed(image: Image.Image):
    model_path = MODEL_DIR / "breed_model.keras"
    labels_path = MODEL_DIR / "breed_labels.json"
    if not model_path.exists() or not labels_path.exists():
        return None, None

    model = load_model(str(model_path))
    labels = load_labels(str(labels_path))
    prediction = model.predict(prepare_image(image), verbose=0)[0]
    top_index = int(np.argmax(prediction))
    return labels[top_index], float(prediction[top_index])


def predict_health(image: Image.Image):
    model_path = MODEL_DIR / "health_model.keras"
    labels_path = MODEL_DIR / "health_labels.json"
    if not model_path.exists() or not labels_path.exists():
        return None, None

    model = load_model(str(model_path))
    labels = load_labels(str(labels_path))
    prediction = model.predict(prepare_image(image), verbose=0)[0]
    score = float(prediction[0]) if prediction.ndim == 1 else float(prediction)
    label = labels[1] if score >= 0.5 else labels[0]
    return label, score


st.set_page_config(
    page_title="Animal AI | Image screening",
    page_icon="🐾",
    layout="centered",
)

st.markdown(
    """
    <style>
    @keyframes drift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes appear {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(196, 224, 199, 0.36), transparent 34%),
            radial-gradient(ellipse at 95% 18%, rgba(242, 222, 181, 0.28), transparent 30%),
            #f8faf6;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.4rem;
        padding-bottom: 3rem;
        max-width: 860px;
        animation: appear 650ms ease-out both;
    }
    .animal-hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.8rem;
        padding: 2.2rem 2.3rem;
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 26px;
        color: #fff;
        background: linear-gradient(115deg, #173f35, #26735a, #4a8d63, #173f35);
        background-size: 260% 260%;
        animation: drift 16s ease infinite;
        box-shadow: 0 18px 44px rgba(27, 74, 56, 0.18);
    }
    .animal-hero::after {
        position: absolute;
        top: -90px;
        right: -40px;
        width: 250px;
        height: 250px;
        border: 1px solid rgba(255, 255, 255, 0.17);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255, 255, 255, 0.035),
                    0 0 0 58px rgba(255, 255, 255, 0.025);
        content: "";
        pointer-events: none;
    }
    .animal-eyebrow {
        margin: 0 0 0.7rem;
        color: #d3ebd8;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .animal-hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #fff;
        font-size: clamp(2rem, 5vw, 3rem);
        letter-spacing: -0.045em;
        line-height: 1.08;
    }
    .animal-hero p:last-child {
        position: relative;
        z-index: 1;
        max-width: 570px;
        margin: 0.85rem 0 0;
        color: #e4f0e7;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    [data-testid="stFileUploader"] {
        padding: 0.35rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.62);
        transition: transform 180ms ease, box-shadow 180ms ease;
    }
    [data-testid="stFileUploader"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(27, 74, 56, 0.09);
    }
    [data-testid="stImage"] img {
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(27, 53, 42, 0.12);
    }
    div[data-testid="stVerticalBlock"] > div:has(> div > [data-testid="stAlert"]) {
        animation: appear 450ms ease-out both;
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="animal-hero">
        <p class="animal-eyebrow">Animal AI · image screening</p>
        <h1>A closer look at animal health.</h1>
        <p>Upload a photo to explore an AI-generated breed estimate and preliminary health indicator.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

breed_ready = any(
    (DATA_DIR / "breed" / "train" / label).glob("*.png")
    for label in ["gir", "sahiwal", "holstein_friesian"]
)
health_ready = any(
    (DATA_DIR / "health" / "train" / label).glob("*.png")
    for label in ["indicator_absent", "indicator_present"]
)
if not breed_ready or not health_ready:
    create_demo_dataset(DATA_DIR)

with st.container(border=True):
    st.subheader("Start with a photo", icon=":material/add_a_photo:")
    st.caption("The current models are trained for cattle only. Choose the animal type before viewing results.")
    uploaded_file = st.file_uploader(
        "Choose an animal image",
        type=["jpg", "jpeg", "png"],
        help="JPG and PNG images are supported.",
    )
    animal_type = st.segmented_control(
        "Animal type",
        options=["Not sure", "Cattle", "Other animal"],
        default="Not sure",
        help="Breed and health models are only trained on cattle images.",
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.space("medium")
    st.image(
        image,
        caption="Your uploaded image",
        alt="Uploaded animal image",
        width="stretch",
    )

    st.header("Screening results", icon=":material/auto_awesome:")
    if animal_type == "Other animal":
        st.warning(
            "The current models are trained only on cattle. Results are unavailable for other animal types.",
            icon=":material/pets:",
        )
    elif animal_type == "Not sure":
        st.info(
            "Choose “Cattle” only if this image shows cattle. The app does not automatically identify animal species.",
            icon=":material/info:",
        )
    else:
        breed_label, breed_score = predict_breed(image)
        health_label, health_score = predict_health(image)
        breed_col, health_col = st.columns(2)

        with breed_col:
            with st.container(border=True, height="stretch"):
                st.markdown(":material/pets: **Cattle breed estimate**")
                if breed_label is not None:
                    st.subheader(breed_label.replace("_", " ").title())
                    st.progress(
                        max(0.0, min(breed_score, 1.0)),
                        text=f"{breed_score * 100:.1f}% model confidence",
                    )
                else:
                    st.info("The breed model is not trained yet.")
                    st.caption("Run `python .\\src\\train_breed.py` to train it.")

        with health_col:
            with st.container(border=True, height="stretch"):
                st.markdown(":material/health_and_safety: **Health indicator**")
                if health_label is not None:
                    display_score = (
                        health_score
                        if health_label == "indicator_present"
                        else 1.0 - health_score
                    )
                    friendly_label = health_label.replace("_", " ").capitalize()
                    st.subheader(friendly_label)
                    st.progress(
                        max(0.0, min(display_score, 1.0)),
                        text=f"{display_score * 100:.1f}% model confidence",
                    )
                else:
                    st.info("The health model is not trained yet.")
                    st.caption("Run `python .\\src\\train_health.py` to train it.")

        st.caption(
            "These model outputs are preliminary screening indicators, not a veterinary diagnosis. "
            "Consult a veterinarian for health concerns."
        )
else:
    st.info(
        "Choose an image above to see its screening results.",
        icon=":material/info:",
    )
