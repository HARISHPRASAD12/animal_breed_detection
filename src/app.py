from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO, StringIO
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from detector import MODEL_PATH, CattleDetector, annotate
from health import PREDICTION_MODEL, classify_crops, SKIN_MODEL, indicator_scores, load_health_model, skin_result

VISIBLE_SIGNS = ("Skin mark or wound", "Visible swelling", "Eye or nasal discharge")
UI_DIR = Path(__file__).resolve().parent / "ui"


@st.cache_resource
def load_detector() -> CattleDetector:
    return CattleDetector()


@st.cache_resource
def cached_health_model(path, modified_ns):
    return load_health_model(path)


def screen_images(path, images):
    return indicator_scores(cached_health_model(path, path.stat().st_mtime_ns), images)


st.set_page_config(
    page_title="Cattle Detection and Health Screening",
    page_icon=":material/document_scanner:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(UI_DIR / "fieldscan.css")
st.html("""
    <section class="fieldscan-hero" aria-labelledby="fieldscan-title">
        <div class="fieldscan-hero__inner">
            <div class="fieldscan-brand"><span aria-hidden="true"></span>Animal<span class="fieldscan-brand__accent">AI</span></div>
            <div class="fieldscan-hero__copy">
                <h1 id="fieldscan-title">Cattle detection<br>and health screening.</h1>
                <p>Upload a photo to locate cattle and document visible health signs for each animal.</p>
                <a class="fieldscan-cta" href="#upload-photo">Upload a photo <span aria-hidden="true">↓</span></a>
            </div>
        </div>
        <aside class="fieldscan-guide" aria-label="Photo tips">
            <h2>For a clearer scan</h2>
            <ul>
                <li>Use even light and avoid deep shadows.</li>
                <li>Keep the animals in view.</li>
                <li>Choose a sharp photo.</li>
            </ul>
        </aside>
    </section>
""")

with st.container(border=True, key="upload-panel"):
    st.header("Upload a cattle photo", anchor="upload-photo")
    uploaded = st.file_uploader(
        "Choose a photo", type=["jpg", "jpeg", "png", "webp"],
        max_upload_size=15, label_visibility="collapsed",
    )
    with st.popover("Detection settings", icon=":material/tune:"):
        threshold = st.slider(
            "Minimum detection score", 0.10, 0.90, 0.35, 0.05,
            help="Lower values may find more cattle and can also increase false detections.",
        )

if uploaded is None:
    st.stop()

try:
    with Image.open(uploaded) as source:
        if source.width * source.height > 25_000_000:
            st.error("Choose a photo with at most 25 megapixels.")
            st.stop()
        image = ImageOps.exif_transpose(source).convert("RGB")
except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
    st.error("The photo could not be opened. Choose a valid JPG, PNG, or WebP image.")
    st.stop()

if not MODEL_PATH.is_file():
    st.error("The model file is missing. Follow the model setup instructions in the project README.")
    st.stop()

image_id = hashlib.sha256(uploaded.getbuffer()).hexdigest()
detection_key = (image_id, threshold, MODEL_PATH.stat().st_mtime_ns)
try:
    if st.session_state.get("detection_key") != detection_key:
        with st.spinner("Detecting cattle..."):
            st.session_state.detections = load_detector().detect(image, threshold)
        st.session_state.detection_key = detection_key
    detections = st.session_state.detections
except cv2.error:
    st.error("The detector could not process this image. Verify the ONNX file against the checksum in the README.")
    st.stop()

result = annotate(image, detections)
with st.container(border=True, key="preview-panel"):
    st.header("Detection preview", anchor=False)
    if detections:
        count = len(detections)
        st.badge(f"{count} {'animal' if count == 1 else 'animals'} detected", color="yellow")
    else:
        st.info("No cattle detected at the selected threshold.")
    st.image(result, width="stretch", alt=f"Uploaded photo with {len(detections)} cattle detections")

if detections:
    st.header("Health screening", anchor=False)
    st.caption("Review visible signs in the photo. Consult a veterinarian for health concerns.")
    predictions = ["Model unavailable"] * len(detections)
    if SKIN_MODEL.is_file():
        try:
            with st.spinner("Screening cattle for visible skin abnormalities..."):
                scores = [
                    score for start in range(0, len(detections), 16)
                    for score in screen_images(
                        SKIN_MODEL, [image.crop(detection.box) for detection in detections[start:start + 16]],
                    )
                ]
            predictions = [skin_result(score) for score in scores]
        except (ImportError, OSError, RuntimeError, ValueError):
            st.warning("Automatic skin screening could not run. You can still record observations.")

    try:
        metadata_path = PREDICTION_MODEL.with_suffix(".json")
        model_version = PREDICTION_MODEL.stat().st_mtime_ns
        estimate_key = (image_id, tuple(item.box for item in detections), model_version, metadata_path.stat().st_mtime_ns)
        if st.session_state.get("estimate_key") != estimate_key:
            with st.spinner("Estimating breed and screening for lumpy skin disease..."):
                model = cached_health_model(PREDICTION_MODEL, model_version)
                metadata = json.loads(metadata_path.read_text())
                breeds = metadata["breed"]["classes"]
                estimates = []
                for start in range(0, len(detections), 16):
                    crops = [image.crop(item.box) for item in detections[start:start + 16]]
                    estimates.extend(classify_crops(model, crops, breeds))
                st.session_state.estimates = estimates
                st.session_state.estimate_key = estimate_key
        estimates = st.session_state.estimates
    except (ImportError, OSError, ValueError, KeyError, RuntimeError):
        st.warning("Breed and disease predictions are unavailable. You can still record and export observations.")
        estimates = [{"Breed estimate": "Unavailable", "Breed score (%)": None,
                      "Lumpy skin screening": "Unavailable", "Lumpy class score (%)": None}
                     for _ in detections]

    if st.session_state.get("review_image") != image_id:
        st.session_state.review_image = image_id
        st.session_state.health_reviews = {}
    selected = st.selectbox(
        "Animal to review", range(len(detections)), format_func=lambda index: f"Cattle {index + 1}",
        key=f"review-{image_id}-{len(detections)}",
    )
    box = detections[selected].box
    reviews = st.session_state.health_reviews
    photo_column, review_column = st.columns([1, 1.25], gap="medium")
    with photo_column:
        st.image(image.crop(box), width="stretch", alt=f"Cropped photo of cattle {selected + 1}")
        estimate = estimates[selected]
        st.write(f"**Breed estimate** · {estimate['Breed estimate']}")
        st.write(f"**Disease screening** · {estimate['Lumpy skin screening']}")
        if estimate["Breed score (%)"] is not None:
            st.caption(f"Breed model score {estimate['Breed score (%)']:.1f}% · "
                       f"Lumpy skin disease class score {estimate['Lumpy class score (%)']:.1f}%")
        st.caption("Research model covering five breeds and one disease. A veterinarian must confirm suspected cases.")
        if predictions[selected] != "Model unavailable":
            st.caption(f"Skin screening — {predictions[selected]}")

    options = ("Unsure or obscured", "Visible", "None observed")
    with review_column.form(f"signs-{image_id}-{box}"):
        previous = reviews.get(box, {})
        observation = {
            sign: st.selectbox(sign, options, index=options.index(previous.get(sign, options[0])))
            for sign in VISIBLE_SIGNS
        }
        if st.form_submit_button("Save observations", type="primary", icon=":material/check:"):
            reviews[box] = observation
            st.success("Observations saved for this photo.")

    st.header("Screening records", anchor=False)
    rows = [
        {"Cattle": index, "Score (%)": round(detection.score * 100, 1),
         "Left": detection.box[0], "Top": detection.box[1],
         "Right": detection.box[2], "Bottom": detection.box[3],
         **estimates[index - 1],
         **({"Automatic skin screening": predictions[index - 1]} if SKIN_MODEL.is_file() else {}),
         **{sign: reviews.get(detection.box, {}).get(sign, "Not reviewed") for sign in VISIBLE_SIGNS}}
        for index, detection in enumerate(detections, 1)
    ]
    st.dataframe(
        rows, hide_index=True, width="stretch",
        column_order=["Cattle", "Score (%)", *estimates[0], *VISIBLE_SIGNS, *(["Automatic skin screening"] if SKIN_MODEL.is_file() else []), "Left", "Top", "Right", "Bottom"],
    )
    csv_output = StringIO()
    writer = csv.DictWriter(csv_output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    image_output = BytesIO()
    result.save(image_output, format="PNG")
    image_column, csv_column = st.columns(2)
    image_column.download_button(
        "Download annotated photo", image_output.getvalue(), "cattle.png", "image/png",
        icon=":material/download:", on_click="ignore", width="stretch",
    )
    csv_column.download_button(
        "Download screening CSV", csv_output.getvalue(), "cattle.csv", "text/csv",
        icon=":material/download:", on_click="ignore", width="stretch",
    )
