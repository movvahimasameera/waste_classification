from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "realwaste-main" / "RealWaste"
MODEL_PATH = PROJECT_DIR / "models" / "waste_classifier.joblib"
IMAGE_SIZE = (128, 128)


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Train the model in waste_classification.ipynb and run its "
            "model-export cell before deploying."
        )

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict):
        raise TypeError("The model artifact must contain a dictionary bundle.")

    model = bundle.get("model")
    scaler = bundle.get("scaler")
    class_names = bundle.get("class_names")
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("The saved classifier must support predict_proba().")
    if not callable(getattr(scaler, "transform", None)):
        raise TypeError("The saved feature scaler must support transform().")
    if not isinstance(class_names, list) or not all(
        isinstance(name, str) for name in class_names
    ):
        raise TypeError("The model artifact must include a list of class names.")
    model_classes = getattr(model, "classes_", None)
    if model_classes is None or len(model_classes) != len(class_names):
        raise ValueError("The saved classifier and class-name list do not match.")
    if any(
        not isinstance(class_index, (int, np.integer))
        or class_index < 0
        or class_index >= len(class_names)
        for class_index in model_classes
    ):
        raise ValueError("The saved classifier contains invalid class indexes.")

    return bundle


def extract_features(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BICUBIC,
    )
    pixels = np.asarray(image)
    gray = rgb2gray(pixels)

    shape_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
    )

    hsv = rgb2hsv(pixels)
    quarters = (
        hsv[:64, :64],
        hsv[:64, 64:],
        hsv[64:, :64],
        hsv[64:, 64:],
    )
    color_features = []
    for quarter in quarters:
        for channel in range(3):
            values = quarter[:, :, channel]
            histogram, _ = np.histogram(
                values,
                bins=8,
                range=(0, 1),
                density=True,
            )
            color_features.extend(histogram)
            color_features.extend((values.mean(), values.std()))

    lbp = local_binary_pattern(
        (gray * 255).astype(np.uint8),
        8,
        1,
        method="uniform",
    )
    texture_features, _ = np.histogram(
        lbp,
        bins=10,
        range=(0, 10),
        density=True,
    )

    return np.concatenate(
        (shape_features, color_features, texture_features)
    ).reshape(1, -1)


def predict_image(
    bundle: dict,
    image: Image.Image,
) -> pd.DataFrame:
    features = extract_features(image)
    scaled_features = bundle["scaler"].transform(features)
    probabilities = bundle["model"].predict_proba(scaled_features)[0]
    class_names = bundle["class_names"]
    model_classes = bundle["model"].classes_

    if len(probabilities) != len(model_classes):
        raise ValueError("The model returned an invalid probability result.")

    predictions = pd.DataFrame(
        {
            "class": [
                class_names[int(class_index)] for class_index in model_classes
            ],
            "probability": probabilities.astype(float),
        }
    )
    return predictions.sort_values("probability", ascending=False)


def get_sample_images(class_name: str) -> list[Path]:
    if not DATASET_DIR.is_dir():
        return []
    return sorted((DATASET_DIR / class_name).glob("*.*"))


def main() -> None:
    st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")
    st.title("♻️ Waste Classification App")
    st.caption("Classify waste images into the categories in the RealWaste dataset.")

    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    class_names = bundle["class_names"]
    st.sidebar.title("Model")
    st.sidebar.write(f"Loaded classifier with {len(class_names)} waste categories.")

    sources = ["Upload an image"]
    if DATASET_DIR.is_dir():
        sources.append("Use a sample image")
    source = st.radio("Choose input type", sources, horizontal=True)

    image = None
    if source == "Upload an image":
        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        selected_class = st.selectbox("Choose a class", class_names)
        sample_files = get_sample_images(selected_class)
        if not sample_files:
            st.warning(f"No sample images were found for {selected_class}.")
            return
        selected_file = st.selectbox(
            "Select a sample image",
            sample_files,
            format_func=lambda path: path.name,
        )
        image = Image.open(selected_file)

    if image is None:
        st.info("Upload or select an image to classify.")
        return

    st.image(image, caption="Image to classify", use_container_width=True)
    if st.button("Predict waste type"):
        predictions = predict_image(bundle, image)
        top_prediction = predictions.iloc[0]
        st.subheader("Prediction")
        st.success(f"Predicted category: {top_prediction['class']}")
        st.write(f"Confidence: {top_prediction['probability'] * 100:.2f}%")
        st.dataframe(predictions, use_container_width=True)


if __name__ == "__main__":
    main()
