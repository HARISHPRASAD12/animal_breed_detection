"""Evaluate cattle detections against image-level bounding-box annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from detector import CattleDetector


def intersection_over_union(first: tuple, second: tuple) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


def evaluate(manifest: Path, threshold: float) -> dict:
    records = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not records:
        raise ValueError("The manifest must contain at least one image.")
    detector = CattleDetector()
    true_positives = false_positives = false_negatives = 0

    for record in records:
        with Image.open(manifest.parent / record["image"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        unmatched = [tuple(box) for box in record["boxes"]]
        for left, top, right, bottom in unmatched:
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError(f"Invalid annotation in {record['image']}")
        predictions = detector.detect(image, threshold)
        for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
            overlaps = [intersection_over_union(prediction.box, box) for box in unmatched]
            best = max(range(len(overlaps)), key=overlaps.__getitem__, default=None)
            if best is not None and overlaps[best] >= 0.5:
                true_positives += 1
                unmatched.pop(best)
            else:
                false_positives += 1
        false_negatives += len(unmatched)

    predicted = true_positives + false_positives
    annotated = true_positives + false_negatives
    return {
        "images": len(records),
        "score_threshold": threshold,
        "iou_threshold": 0.5,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / predicted if predicted else None,
        "recall": true_positives / annotated if annotated else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    print(json.dumps(evaluate(args.manifest, args.threshold), indent=2))
