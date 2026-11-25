from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SKIN_MODEL = ROOT / "models" / "cattle_skin_screening.keras"
IMAGE_SIZE = (224, 224)


def load_health_model(path: Path):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def indicator_scores(model, images: list[Image.Image]) -> list[float]:
    import tensorflow as tf

    batch = tf.stack([tf.image.resize(np.asarray(image.convert("RGB")), IMAGE_SIZE) for image in images])
    scores = np.asarray(model(batch, training=False)).reshape(-1)
    if len(scores) != len(images) or not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
        raise ValueError("Invalid health model output")
    return scores.tolist()


def skin_result(score: float) -> str:
    if score >= 0.6:
        return "Possible visible skin abnormality"
    if score <= 0.4:
        return "No visible skin abnormality flagged"
    return "Inconclusive"

PREDICTION_MODEL = ROOT / "models" / "cattle_predictions.keras"


def pixels(photo):
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    return np.asarray(photo.resize(IMAGE_SIZE, Image.Resampling.BILINEAR))


def classify_crops(model, crops, breeds):
    output = model(np.stack([pixels(crop) for crop in crops]), training=False)
    results = []
    breed_matrix = np.asarray(output["breed"])
    disease_scores = np.asarray(output["disease"])
    if breed_matrix.shape != (len(crops), len(breeds)) or disease_scores.shape != (len(crops), 1):
        raise ValueError("Prediction dimensions do not match the cattle crops and breed labels.")
    for scores in (breed_matrix, disease_scores):
        if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
            raise ValueError("Invalid model score.")
    for breed_scores, disease in zip(breed_matrix, disease_scores[:, 0]):
        index = int(np.argmax(breed_scores))
        results.append({
            "Breed estimate": breeds[index],
            "Breed score (%)": round(float(breed_scores[index]) * 100, 1),
            "Lumpy skin screening": "Possible lumpy skin disease" if disease >= 0.5 else "Lumpy skin disease not flagged",
            "Lumpy class score (%)": round(float(disease) * 100, 1),
        })
    return results
