from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

IMAGE_SIZE = (224, 224)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_breed_sprite(label: str, index: int, width: int = 224, height: int = 224) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    cx, cy = width // 2, height // 2

    palette = {
        "gir": (188, 160, 120),
        "sahiwal": (170, 146, 104),
        "holstein_friesian": (225, 225, 225),
    }
    base_color = palette.get(label, (200, 200, 200))

    draw.ellipse((cx - 86, cy - 40, cx + 86, cy + 52), fill=base_color)
    draw.ellipse((cx - 40, cy - 72, cx + 40, cy - 10), fill=base_color)
    draw.rectangle((cx - 38, cy + 52, cx + 38, cy + 110), fill=base_color)
    draw.rectangle((cx - 120, cy + 50, cx - 88, cy + 110), fill=base_color)
    draw.rectangle((cx + 88, cy + 50, cx + 120, cy + 110), fill=base_color)

    for i in range(7):
        x = cx - 70 + (i % 4) * 24 + (index % 10)
        y = cy - 26 + ((i // 2) * 18) + (index % 8)
        draw.ellipse((x, y, x + 16, y + 16), fill=(80, 80, 80))

    if label == "holstein_friesian":
        draw.line((cx - 28, cy - 32, cx + 26, cy - 32), fill=(70, 70, 70), width=6)
    if label == "gir":
        draw.rectangle((cx - 10, cy - 88, cx + 10, cy - 54), fill=(100, 74, 52))
    if label == "sahiwal":
        draw.polygon([(cx - 10, cy - 90), (cx + 10, cy - 90), (cx, cy - 120)], fill=(100, 74, 52))

    return img


def make_health_sprite(label: str, index: int, width: int = 224, height: int = 224) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    if label == "indicator_absent":
        draw.ellipse((60, 64, 160, 164), fill=(135, 190, 140))
        draw.rectangle((78, 120, 142, 152), fill=(120, 150, 120))
        draw.line((92, 80, 128, 96), fill=(95, 120, 95), width=6)
    else:
        draw.ellipse((60, 64, 160, 164), fill=(160, 160, 160))
        draw.ellipse((96, 92, 130, 124), fill=(190, 80, 80))
        draw.ellipse((84, 110, 118, 146), fill=(222, 108, 108))
        draw.rectangle((72, 165, 148, 175), fill=(220, 220, 220))

    return img


def create_demo_dataset(root_dir: str | Path = DATA_DIR) -> dict[str, list[str]]:
    root_dir = Path(root_dir)
    breed_classes = ["gir", "sahiwal", "holstein_friesian"]
    health_classes = ["indicator_absent", "indicator_present"]

    for split in ["train", "val", "test"]:
        for label in breed_classes:
            ensure_dir(root_dir / "breed" / split / label)
        for label in health_classes:
            ensure_dir(root_dir / "health" / split / label)

    for label in breed_classes:
        for split, count in {"train": 24, "val": 12, "test": 12}.items():
            for idx in range(count):
                image = make_breed_sprite(label, idx)
                image.save(root_dir / "breed" / split / label / f"{label}_{idx}.png")

    for label in health_classes:
        for split, count in {"train": 24, "val": 12, "test": 12}.items():
            for idx in range(count):
                image = make_health_sprite(label, idx)
                image.save(root_dir / "health" / split / label / f"{label}_{idx}.png")

    return {"breed": breed_classes, "health": health_classes}


def get_class_names(data_dir: str | Path = DATA_DIR, dataset_name: str = "breed") -> list[str]:
    root = Path(data_dir) / dataset_name / "train"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_image_dataset(data_dir: str | Path, dataset_name: str, batch_size: int = 16, image_size: tuple[int, int] = IMAGE_SIZE):
    root = Path(data_dir) / dataset_name
    train_ds = tf.keras.utils.image_dataset_from_directory(
        root / "train",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root / "val",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        root / "test",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    return train_ds, val_ds, test_ds


def save_class_names(class_names: list[str], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)


def load_class_names(save_path: str | Path) -> list[str]:
    with open(save_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def preprocess_single_image(image: np.ndarray, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    image = tf.keras.utils.load_img(image, target_size=target_size)
    arr = tf.keras.utils.img_to_array(image)
    arr = arr.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)
