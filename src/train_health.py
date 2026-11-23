from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import tensorflow as tf

from health import IMAGE_SIZE, ROOT, SKIN_MODEL

CLASSES = ["no_visible_abnormality", "visible_abnormality"]
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def validate_splits(root: Path) -> None:
    seen = {}
    for split in SPLITS:
        for label in CLASSES:
            files = [
                path for path in (root / split / label).rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
            ]
            if not files:
                raise ValueError(f"No labeled images in {root / split / label}")
            for path in files:
                checksum = hashlib.sha256(path.read_bytes()).digest()
                group = (split, label)
                if checksum in seen and seen[checksum] != group:
                    raise ValueError(f"Duplicate image across splits or labels in {path}")
                seen[checksum] = group


def train(root: Path, epochs: int) -> dict[str, float]:
    validate_splits(root)
    if root.resolve().is_relative_to(ROOT / "data" / "health"):
        raise ValueError("Use a separate labeled cattle-photo dataset.")

    datasets = {
        split: tf.keras.utils.image_dataset_from_directory(
            root / split, class_names=CLASSES, label_mode="binary",
            image_size=IMAGE_SIZE, batch_size=16, shuffle=split == "train", seed=42,
        )
        for split in SPLITS
    }
    tf.keras.utils.set_random_seed(42)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, pooling="avg", weights="imagenet",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    normalized = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    features = base(normalized, training=False)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="roc_auc"),
    ])
    model.fit(
        datasets["train"], validation_data=datasets["val"], epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    results = model.evaluate(datasets["test"], return_dict=True, verbose=0)
    with TemporaryDirectory(dir=SKIN_MODEL.parent) as folder:
        candidate = Path(folder) / SKIN_MODEL.name
        model.save(candidate)
        candidate.replace(SKIN_MODEL)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a research model for visible cattle skin abnormalities.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    try:
        metrics = train(args.dataset, args.epochs)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {SKIN_MODEL}")
