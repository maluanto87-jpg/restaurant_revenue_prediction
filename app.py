"""
Restaurant Revenue Class Prediction — Streamlit App
Ported from Restaurant_Revenue_prediction.ipynb

This version loads the artifacts the notebook exports at the end:
  - restaurant_model.pkl  (trained Random Forest classifier)
  - scaler.pkl            (StandardScaler fit on the training features)
  - encoders.pkl          (dict of {column_name: LabelEncoder})

Place these three files next to app.py (e.g. in the repo root) and the app
will load them automatically. If they aren't found, the sidebar lets you
upload them manually.

Note: Random Forest (like Decision Tree) is a tree-based model, so it was
trained on the raw, unscaled features. The app detects this automatically
(see SCALED_MODEL_TYPES below) and skips scaling at inference time.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.set_page_config(
    page_title="Restaurant Revenue Class Predictor",
    page_icon="🍽️",
    layout="wide",
)

APP_DIR = Path(__file__).parent

# Feature order exactly as produced by the notebook:
# df.drop(["Name"], axis=1) -> encode categoricals -> X = df.drop(["Revenue", "Revenue_Class"], axis=1)
FEATURE_COLUMNS = [
    "Location",
    "Cuisine",
    "Rating",
    "Seating Capacity",
    "Average Meal Price",
    "Marketing Budget",
    "Social Media Followers",
    "Chef Experience Years",
    "Number of Reviews",
    "Avg Review Length",
    "Ambience Score",
    "Service Quality Score",
    "Parking Availability",
    "Weekend Reservations",
    "Weekday Reservations",
]

CATEGORICAL_COLUMNS = ["Location", "Cuisine", "Parking Availability"]

# Sensible defaults/ranges taken from the dataset's descriptive statistics,
# used only to pre-fill and guide the numeric input widgets.
NUMERIC_FIELD_CONFIG = {
    "Rating": dict(min_value=1.0, max_value=5.0, value=4.0, step=0.1, is_float=True),
    "Seating Capacity": dict(min_value=1, max_value=500, value=60, step=1, is_float=False),
    "Average Meal Price": dict(min_value=1.0, max_value=500.0, value=48.0, step=0.5, is_float=True),
    "Marketing Budget": dict(min_value=0, max_value=50000, value=3200, step=50, is_float=False),
    "Social Media Followers": dict(min_value=0, max_value=500000, value=36000, step=100, is_float=False),
    "Chef Experience Years": dict(min_value=0, max_value=50, value=10, step=1, is_float=False),
    "Number of Reviews": dict(min_value=0, max_value=5000, value=520, step=1, is_float=False),
    "Avg Review Length": dict(min_value=0.0, max_value=1000.0, value=175.0, step=1.0, is_float=True),
    "Ambience Score": dict(min_value=0.0, max_value=10.0, value=5.5, step=0.1, is_float=True),
    "Service Quality Score": dict(min_value=0.0, max_value=10.0, value=5.5, step=0.1, is_float=True),
    "Weekend Reservations": dict(min_value=0, max_value=500, value=29, step=1, is_float=False),
    "Weekday Reservations": dict(min_value=0, max_value=500, value=29, step=1, is_float=False),
}

# Models that were trained on the *scaled* features in the notebook.
SCALED_MODEL_TYPES = (LogisticRegression, KNeighborsClassifier, SVC)


# --------------------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------------------
def _load_pickle(source) -> object:
    if hasattr(source, "read"):  # uploaded file
        return pickle.load(source)
    with open(source, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_local_artifacts():
    model_path = APP_DIR / "restaurant_model.pkl"
    scaler_path = APP_DIR / "scaler.pkl"
    encoders_path = APP_DIR / "encoders.pkl"

    if model_path.exists() and scaler_path.exists() and encoders_path.exists():
        model = _load_pickle(model_path)
        scaler = _load_pickle(scaler_path)
        encoders = _load_pickle(encoders_path)
        return model, scaler, encoders
    return None, None, None


def get_artifacts():
    model, scaler, encoders = load_local_artifacts()

    if model is not None:
        st.sidebar.success("✅ Loaded model, scaler, and encoders from the app folder.")
        return model, scaler, encoders

    st.sidebar.warning(
        "Couldn't find `restaurant_model.pkl`, `scaler.pkl`, and `encoders.pkl` "
        "next to app.py. Upload them below."
    )
    model_file = st.sidebar.file_uploader("restaurant_model.pkl", type=["pkl"])
    scaler_file = st.sidebar.file_uploader("scaler.pkl", type=["pkl"])
    encoders_file = st.sidebar.file_uploader("encoders.pkl", type=["pkl"])

    if not (model_file and scaler_file and encoders_file):
        st.info("👈 Upload all three `.pkl` files in the sidebar to get started.")
        st.stop()

    model = _load_pickle(model_file)
    scaler = _load_pickle(scaler_file)
    encoders = _load_pickle(encoders_file)
    return model, scaler, encoders


# --------------------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Encode categorical columns and enforce the training column order."""
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = encoders[col]
            unseen = set(df[col].astype(str)) - set(le.classes_)
            if unseen:
                raise ValueError(
                    f"Column '{col}' has value(s) {sorted(unseen)} the model has "
                    f"never seen. Known values: {list(le.classes_)}"
                )
            df[col] = le.transform(df[col].astype(str))
    return df[FEATURE_COLUMNS]


def predict(df: pd.DataFrame, model, scaler):
    needs_scaling = isinstance(model, SCALED_MODEL_TYPES)
    X = scaler.transform(df) if needs_scaling else df
    preds = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return preds, proba


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("🍽️ Restaurant Revenue Class Predictor")

st.caption(
    "Predicts whether a restaurant falls into the **High** or **Low** revenue class, "
    "using the model exported from the analysis notebook."
)

with st.sidebar:
    st.header("Model Artifacts")

model, scaler, encoders = get_artifacts()

model_name = type(model).__name__
needs_scaling = isinstance(model, SCALED_MODEL_TYPES)

# Display loaded model on the main page
st.success(f"Loaded Model: {model_name}")

with st.sidebar:
    st.markdown(f"**Loaded Model:** `{model_name}`")
    st.markdown(
        f"**Uses scaler at inference:** {'Yes' if needs_scaling else 'No'}"
    )

tab_single, tab_batch, tab_importance = st.tabs(
    ["🔮 Single Prediction", "📄 Batch Prediction", "🌲 Feature Importance"]
)

# --------------------------------------------------------------------------------------
# Single prediction tab
# --------------------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter restaurant details")

    col1, col2, col3 = st.columns(3)
    user_input = {}

    for i, col in enumerate(FEATURE_COLUMNS):
        target_col = [col1, col2, col3][i % 3]
        with target_col:
            if col in CATEGORICAL_COLUMNS:
                options = list(encoders[col].classes_)
                user_input[col] = st.selectbox(col, options=options)
            else:
                cfg = NUMERIC_FIELD_CONFIG[col]
                if cfg["is_float"]:
                    user_input[col] = st.number_input(
                        col,
                        min_value=float(cfg["min_value"]),
                        max_value=float(cfg["max_value"]),
                        value=float(cfg["value"]),
                        step=float(cfg["step"]),
                    )
                else:
                    user_input[col] = st.number_input(
                        col,
                        min_value=int(cfg["min_value"]),
                        max_value=int(cfg["max_value"]),
                        value=int(cfg["value"]),
                        step=int(cfg["step"]),
                    )

    st.divider()
    if st.button("Predict Revenue Class", type="primary"):
        input_df = pd.DataFrame([user_input])[FEATURE_COLUMNS]
        try:
            processed = preprocess(input_df, encoders)
            preds, proba = predict(processed, model, scaler)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
        else:
            prediction = preds[0]
            label = "High Revenue 💰" if prediction == 1 else "Low Revenue 📉"

            st.subheader("Result")
            st.metric("Predicted Revenue Class", label)

            if proba is not None:
                # Display probabilities as percentages
                st.subheader("Prediction Probability")
                st.write(f"🔴 Low Revenue (0): {proba[0][0]*100:.2f}%")
                st.write(f"🟢 High Revenue (1): {proba[0][1]*100:.2f}%")
                # Bar chart
                proba_df = pd.DataFrame({
                    "Class": ["Low Revenue (0)", "High Revenue (1)"],
                    "Probability": [proba[0][0], proba[0][1]]
                })
                st.bar_chart(proba_df.set_index("Class"))

# --------------------------------------------------------------------------------------
# Batch prediction tab
# --------------------------------------------------------------------------------------
with tab_batch:
    st.subheader("Predict for many restaurants at once")
    st.caption(
        "Upload a CSV with the same feature columns used in training "
        "(Name and Revenue/Revenue_Class columns are optional and will be ignored)."
    )
    st.code(", ".join(FEATURE_COLUMNS), language=None)

    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing:
            st.error(f"The uploaded CSV is missing required columns: {missing}")
        else:
            try:
                processed = preprocess(batch_df, encoders)
                preds, proba = predict(processed, model, scaler)
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
            else:
                result_df = batch_df.copy()
                result_df["Predicted_Revenue_Class"] = preds
                result_df["Predicted_Label"] = np.where(
                    preds == 1, "High Revenue", "Low Revenue"
                )
                if proba is not None:
                    result_df["Probability_High"] = proba[:, 1]

                st.success(f"Predicted {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=csv_bytes,
                    file_name="restaurant_revenue_predictions.csv",
                    mime="text/csv",
                )

# --------------------------------------------------------------------------------------
# Feature importance tab
# --------------------------------------------------------------------------------------
with tab_importance:
    st.subheader("Feature importance")
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)
        st.bar_chart(fi.set_index("Feature")["Importance"])
        st.dataframe(fi, use_container_width=True)
    elif hasattr(model, "coef_"):
        fi = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Coefficient": model.coef_[0]}
        ).sort_values("Coefficient", key=abs, ascending=False)
        st.bar_chart(fi.set_index("Feature")["Coefficient"])
        st.dataframe(fi, use_container_width=True)
    else:
        st.info(f"`{model_name}` doesn't expose feature importances or coefficients.")
