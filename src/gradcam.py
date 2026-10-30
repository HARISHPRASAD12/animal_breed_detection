from __future__ import annotations

import numpy as np
import tensorflow as tf
from PIL import Image


def grad_cam_heatmap(model: tf.keras.Model, image: np.ndarray, last_conv_layer_name: str, pred_index: int | None = None) -> np.ndarray:
    img_array = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, outputs = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(outputs[0]))
        class_channel = outputs[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs @ pooled_grads
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = Image.fromarray(heatmap, mode="L").resize((image.shape[1], image.shape[0]))
    heatmap_array = np.asarray(heatmap_color)
    heatmap_array = np.stack([heatmap_array, heatmap_array, heatmap_array], axis=-1)

    blended = image.astype(np.float32) * (1.0 - alpha) + heatmap_array.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
