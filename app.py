from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
except ImportError:
    tf = None
    MobileNetV2 = None
    Model = None
    layers = None

DATASET_DIR = Path("realwaste-main/RealWaste")
MODEL_PATH = Path("models/waste_classifier.keras")
IMAGE_SIZE = (224, 224)
MAX_EPOCHS = 8


@st.cache_data
def get_class_names() -> list[str]:
    if not DATASET_DIR.exists():
        return []
    return sorted(
        entry.name for entry in DATASET_DIR.iterdir() if entry.is_dir()
    )


@st.cache_resource
def get_model(class_names: list[str]):
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Install it with `pip install tensorflow`.")

    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "Dataset folder not found. Expected 'realwaste-main/RealWaste'."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=32,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=32,
        label_mode="categorical",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    return model


@st.cache_data
def get_sample_images(class_names: list[str]):
    sample_paths = []
    for class_name in class_names:
        class_dir = DATASET_DIR / class_name
        images = sorted(class_dir.glob("*.*"))
        if images:
            sample_paths.append(images[0])
    return sample_paths


def prepare_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.fit(image.convert("RGB"), (IMAGE_SIZE[0], IMAGE_SIZE[1]), method=Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_single(model, image: Image.Image, class_names: list[str]):
    prepared = prepare_image(image)
    probs = model.predict(prepared, verbose=0)[0]
    predictions = pd.DataFrame(
        {
            "class": class_names,
            "probability": [float(p) for p in probs],
        }
    ).sort_values("probability", ascending=False)
    top_prediction = predictions.iloc[0]
    return predictions, top_prediction


def main() -> None:
    st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")
    st.title("♻️ Waste Classification App")
    st.caption("Classify waste images into the categories in the RealWaste dataset.")

    class_names = get_class_names()
    if not class_names:
        st.error(
            "The dataset folder was not found. Please ensure 'realwaste-main/RealWaste' exists in the project root."
        )
        st.stop()

    st.sidebar.title("Training and model")
    st.sidebar.write(f"Detected classes: {len(class_names)}")
    st.sidebar.write(", ".join(class_names))
    st.sidebar.markdown("---")
    st.sidebar.write("The app will train a lightweight MobileNetV2 model on first run if no saved model is available.")

    try:
        model = get_model(class_names)
    except Exception as exc:
        st.error(f"Model setup failed: {exc}")
        st.stop()

    source = st.radio("Choose input type", ["Upload an image", "Use a sample image"], horizontal=True)

    if source == "Upload an image":
        uploaded_file = st.file_uploader("Upload a waste image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_container_width=True)
            clicked = st.button("Predict waste type")
            if clicked:
                predictions, top_prediction = predict_single(model, image, class_names)
                st.subheader("Prediction")
                st.success(f"Predicted category: {top_prediction['class']}")
                st.write(f"Confidence: {top_prediction['probability'] * 100:.2f}%")
                st.dataframe(predictions.head(9), use_container_width=True)
        else:
            st.info("Please upload an image to classify.")
    else:
        sample_paths = get_sample_images(class_names)
        if not sample_paths:
            st.warning("No sample images were found in the dataset.")
            st.stop()

        selected_label = st.selectbox("Choose a class", class_names)
        class_dir = DATASET_DIR / selected_label
        sample_files = sorted(class_dir.glob("*.*"))
        selected_file = st.selectbox("Select a sample image", sample_files, format_func=lambda p: p.name)
        image = Image.open(selected_file)
        st.image(image, caption=f"Sample from {selected_label}", use_container_width=True)

        if st.button("Predict sample"):
            predictions, top_prediction = predict_single(model, image, class_names)
            st.subheader("Prediction")
            st.success(f"Predicted category: {top_prediction['class']}")
            st.write(f"Confidence: {top_prediction['probability'] * 100:.2f}%")
            st.dataframe(predictions.head(9), use_container_width=True)


if __name__ == "__main__":
    main()
