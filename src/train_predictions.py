"""Train breed and lumpy-skin image classifiers on the two local Kaggle archives."""
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from health import ROOT, IMAGE_SIZE, pixels


def train_head(base, archive, name):
    seen, features, labels, batch = set(), [], [], []

    def flush():
        if batch:
            features.extend(base(np.asarray(batch, dtype="float32") / 127.5 - 1, training=False).numpy())
            batch.clear()

    with ZipFile(archive) as data:
        for filename in sorted(data.namelist()):
            path = Path(filename)
            if path.suffix.lower() not in {".jpeg", ".jpg", ".png"} or path.stem.endswith("_c"):
                continue
            try:
                with Image.open(BytesIO(data.read(filename))) as photo:
                    image = pixels(photo)
            except (UnidentifiedImageError, OSError):
                continue
            digest = hashlib.sha256(image.tobytes()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            labels.append(path.parent.name.removesuffix(" cattle"))
            batch.append(image)
            if len(batch) == 16:
                flush()
        flush()

    x, y = np.asarray(features), np.asarray(labels)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    classifier = LogisticRegression(max_iter=1500, class_weight="balanced").fit(x[train], y[train])
    prediction = classifier.predict(x[test])
    classes = classifier.classes_.tolist()
    metrics = dict(classes=classes, train=len(train), test=len(test),
                   accuracy=accuracy_score(y[test], prediction),
                   balanced_accuracy=balanced_accuracy_score(y[test], prediction),
                   confusion_matrix=confusion_matrix(y[test], prediction, labels=classes).tolist())
    print(name, metrics, flush=True)
    return classifier, metrics


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    base = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, pooling="avg")
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = base(tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs), training=False)
    outputs, report = {}, {}
    for name, archive in (("breed", "cattle-breeds.zip"), ("disease", "cow-lumpy.zip")):
        classifier, report[name] = train_head(base, ROOT / "data" / archive, name)
        binary = len(classifier.classes_) == 2
        head = tf.keras.layers.Dense(classifier.coef_.shape[0], activation="sigmoid" if binary else "softmax", name=name)
        outputs[name] = head(features)
        head.set_weights([classifier.coef_.T, classifier.intercept_])
    tf.keras.Model(inputs, outputs).save(ROOT / "models/cattle_predictions.keras")
    (ROOT / "models/cattle_predictions.json").write_text(json.dumps(report, indent=2))
