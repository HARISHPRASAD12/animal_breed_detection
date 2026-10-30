from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from preprocessing import DATA_DIR, IMAGE_SIZE, create_demo_dataset, load_image_dataset, save_class_names

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_health_model() -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def train_health() -> tuple[str, list[str]]:
    health_dir = DATA_DIR / "health"
    has_images = any((health_dir / "train" / label).glob("*.png") for label in ["indicator_absent", "indicator_present"])
    if not has_images:
        create_demo_dataset(DATA_DIR)

    train_ds, val_ds, test_ds = load_image_dataset(DATA_DIR, "health", batch_size=16)
    class_names = train_ds.class_names
    model = build_health_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    save_class_names(class_names, MODEL_DIR / "health_labels.json")
    model_path = MODEL_DIR / "health_model.keras"
    model.save(model_path)
    print(f"Health model saved to {model_path}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return str(model_path), class_names


if __name__ == "__main__":
    train_health()
