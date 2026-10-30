# Waste Classification

This Streamlit app predicts one of the nine RealWaste categories using the
scikit-learn ensemble in `waste_classification.ipynb`. The app only loads the
trained model; it does not train a model or load the dataset into memory at
startup.

## Create the model artifact

1. Install the project dependencies with `pip install -r requirements.txt`.
2. Open `waste_classification.ipynb` from the project root and run its cells in
   order. Training can take time and should be done locally or in a notebook
   environment, not on Streamlit Community Cloud.
3. Run the **model export** cell immediately after the model-fitting cell. It
   writes `models/waste_classifier.joblib`.
4. Keep that artifact alongside the app when deploying. If it is small enough,
   commit it to the repository. GitHub rejects files larger than 100 MB; use
   Git LFS or an external model host for a larger artifact.

The serialized model must be loaded with the same scikit-learn version used to
create it. This project pins the required version in `requirements.txt`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the model artifact is missing, the app displays an actionable error instead
of attempting to train on startup. The sample-image picker is available when
`realwaste-main/RealWaste` is present; image uploads do not require the dataset.
