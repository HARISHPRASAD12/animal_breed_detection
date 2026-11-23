from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "object_detection_yolox_2022nov.onnx"
INPUT_SIZE = 640
COW_CLASS = 19  # Zero-indexed COCO class in the OpenCV Zoo export.


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    score: float


class CattleDetector:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.lock = Lock()
        grids = []
        strides = []
        for stride in (8, 16, 32):
            axis = np.arange(INPUT_SIZE // stride)
            x, y = np.meshgrid(axis, axis)
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full((x.size, 1), stride))
        self.grid = np.concatenate(grids)
        self.strides = np.concatenate(strides)

    def detect(self, image: Image.Image, threshold: float = 0.35) -> list[Detection]:
        width, height = image.size
        scale = INPUT_SIZE / max(width, height)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        resized = cv2.resize(np.asarray(image.convert("RGB")), (resized_width, resized_height))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.float32)
        padded[:resized_height, :resized_width] = resized

        # The exported network expects RGB pixels in the original 0–255 range.
        with self.lock:
            self.net.setInput(cv2.dnn.blobFromImage(padded))
            predictions = self.net.forward()[0].copy()

        classes = predictions[:, 5:]
        scores = predictions[:, 4] * classes[:, COW_CLASS]
        keep = (classes.argmax(axis=1) == COW_CLASS) & (scores > threshold)
        rows, scores = predictions[keep], scores[keep]
        if not len(rows):
            return []

        centers = (rows[:, :2] + self.grid[keep]) * self.strides[keep]
        sizes = np.exp(rows[:, 2:4]) * self.strides[keep]
        image_scale = np.array([resized_width / width, resized_height / height])
        top_left = np.clip((centers - sizes / 2) / image_scale, 0, [width, height])
        bottom_right = np.clip((centers + sizes / 2) / image_scale, 0, [width, height])
        valid = np.all(bottom_right - top_left >= 1, axis=1)
        top_left, bottom_right, scores = top_left[valid], bottom_right[valid], scores[valid]
        boxes = np.column_stack((top_left, bottom_right - top_left))
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, 0.45)
        return [
            Detection(tuple(np.rint([*top_left[i], *bottom_right[i]]).astype(int).tolist()), float(scores[i]))
            for i in indices
        ]


def annotate(image: Image.Image, detections: list[Detection]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=max(12, min(48, max(image.size) // 50)))
    green, yellow, ink = "#009B4D", "#FFCC00", "#193326"
    for number, detection in enumerate(detections, 1):
        left, top, right, bottom = detection.box
        bounds = (left, top, right - 1, bottom - 1)
        stroke = max(2, font.size // 5)
        draw.rectangle(bounds, outline=ink, width=stroke + 2)
        draw.rectangle(bounds, outline=green, width=stroke)
        label = f"Cattle {number}  {detection.score:.0%}"
        text_box = draw.textbbox((0, 0), label, font=font, anchor="lt")
        label_width, label_height = text_box[2] + 8, text_box[3] + 8
        x = max(0, min(left, image.width - label_width))
        y = max(0, top - label_height)
        draw.rectangle((x, y, x + label_width, y + label_height), fill=yellow, outline=ink)
        draw.text((x + 4, y + 4), label, fill=ink, font=font, anchor="lt")
    return output
