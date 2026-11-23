from pathlib import Path

import numpy as np
from PIL import Image

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
