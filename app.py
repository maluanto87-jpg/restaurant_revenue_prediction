"""
Restaurant Revenue Class Prediction — Streamlit App
Ported from Restaurant_Revenue_prediction.ipynb

Pipeline (mirrors the notebook):
  1. Load restaurant_classification.csv
  2. Drop "Name", label-encode categorical columns
  3. Split features (X) / target (y = Revenue_Class)
  4. Train/test split (80/20, stratified)
  5. Scale features for models that need it
  6. Train Logistic Regression, Decision Tree, Random Forest, KNN, SVM
  7. Pick the best model by accuracy and use it for live predictions
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Restaurant Revenue Class Predictor",
    page_icon="🍽️",
    layout="wide",
)

TARGET_COL = "Revenue_Class"
DROP_COLS = ["Name", "Revenue"]  # Revenue is the raw numeric value; Revenue_Class is the target

# Models that were trained on the *scaled* features in the notebook.
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM"}


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


# --------------------------------------------------------------------------------------
# Training pipeline (cached so it only runs once per dataset)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models...")
def train_pipeline(df: pd.DataFrame):
    df = df.copy()

    # Drop columns the notebook drops / doesn't use as features
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset is missing the target column '{TARGET_COL}'.")

    # Encode categorical columns, keep the encoders so the UI can use them later
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", random_state=42, probability=True),
    }

    trained_models = {}
    rows = []
    confusion_matrices = {}

    for name, model in model_defs.items():
        use_scaled = name in SCALED_MODELS
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        model.fit(Xtr, y_train)
        pred = model.predict(Xte)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1 Score": f1_score(y_test, pred),
            }
        )
        confusion_matrices[name] = confusion_matrix(y_test, pred)
        trained_models[name] = model

    results = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    best_model_name = results.iloc[0]["Model"]

    feature_importance = None
    if "Random Forest" in trained_models:
        feature_importance = pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": trained_models["Random Forest"].feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False)

    return {
        "models": trained_models,
        "encoders": encoders,
        "scaler": scaler,
        "feature_columns": list(X.columns),
        "categorical_columns": categorical_columns,
        "results": results,
        "best_model_name": best_model_name,
        "confusion_matrices": confusion_matrices,
        "X": X,
    }


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("🍽️ Restaurant Revenue Class Predictor")
st.caption(
    "Predicts whether a restaurant falls into the **High** or **Low** revenue class, "
    "based on the model pipeline from the original analysis notebook."
)

with st.sidebar:
    st.header("1. Dataset")
    uploaded_file = st.file_uploader(
        "Upload restaurant_classification.csv", type=["csv"]
    )
    st.caption(
        "Upload the same CSV used in the notebook "
        "(`restaurant_classification.csv`). The app trains fresh models on it."
    )

if uploaded_file is None:
    st.info("👈 Upload the `restaurant_classification.csv` dataset to get started.")
    st.stop()

try:
    raw_df = load_data(uploaded_file)
    pipeline = train_pipeline(raw_df)
except Exception as e:  # noqa: BLE001
    st.error(f"Something went wrong while loading/training on this dataset: {e}")
    st.stop()

models = pipeline["models"]
encoders = pipeline["encoders"]
scaler = pipeline["scaler"]
feature_columns = pipeline["feature_columns"]
categorical_columns = pipeline["categorical_columns"]
results = pipeline["results"]
best_model_name = pipeline["best_model_name"]
confusion_matrices = pipeline["confusion_matrices"]
X = pipeline["X"]

tab_predict, tab_performance, tab_data = st.tabs(
    ["🔮 Predict", "📊 Model Performance", "🗂️ Data Overview"]
)

# --------------------------------------------------------------------------------------
# Predict tab
# --------------------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter restaurant details")

    with st.sidebar:
        st.header("2. Choose model")
        model_name = st.selectbox(
            "Model used for prediction",
            options=list(models.keys()),
            index=list(models.keys()).index(best_model_name),
        )
        st.success(f"🏆 Best model on test data: **{best_model_name}**")

    col1, col2, col3 = st.columns(3)
    user_input = {}

    for i, col in enumerate(feature_columns):
        target_col = [col1, col2, col3][i % 3]
        with target_col:
            if col in categorical_columns:
                options = list(encoders[col].classes_)
                user_input[col] = st.selectbox(col, options=options)
            else:
                col_data = X[col]
                is_float = np.issubdtype(col_data.dtype, np.floating)
                min_v = float(col_data.min())
                max_v = float(col_data.max())
                mean_v = float(col_data.mean())
                if is_float:
                    user_input[col] = st.number_input(
                        col, value=round(mean_v, 2), min_value=None, max_value=None, step=0.1
                    )
                else:
                    user_input[col] = st.number_input(
                        col, value=int(round(mean_v)), step=1
                    )

    st.divider()
    if st.button("Predict Revenue Class", type="primary"):
        input_df = pd.DataFrame([user_input])[feature_columns]

        # Apply the same label encoders used during training
        for col in categorical_columns:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

        model = models[model_name]
        if model_name in SCALED_MODELS:
            input_features = scaler.transform(input_df)
        else:
            input_features = input_df

        prediction = model.predict(input_features)[0]
        label = "High Revenue 💰" if prediction == 1 else "Low Revenue 📉"

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_features)[0]

        st.subheader("Result")
        st.metric("Predicted Revenue Class", label)

        if proba is not None:
            proba_df = pd.DataFrame(
                {"Class": ["Low Revenue (0)", "High Revenue (1)"], "Probability": proba}
            )
            st.bar_chart(proba_df.set_index("Class"))

# --------------------------------------------------------------------------------------
# Model performance tab
# --------------------------------------------------------------------------------------
with tab_performance:
    st.subheader("Model comparison")
    st.dataframe(
        results.style.format(
            {"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"}
        ),
        use_container_width=True,
    )
    st.bar_chart(results.set_index("Model")["Accuracy"])

    st.subheader("Confusion matrix")
    cm_model = st.selectbox(
        "Model", options=list(confusion_matrices.keys()), key="cm_model"
    )
    cm = confusion_matrices[cm_model]
    cm_df = pd.DataFrame(
        cm, index=["Actual: Low", "Actual: High"], columns=["Pred: Low", "Pred: High"]
    )
    st.dataframe(cm_df, use_container_width=True)

    if pipeline.get("feature_importance") is not None:
        st.subheader("Feature importance (Random Forest)")
        st.bar_chart(pipeline["feature_importance"].set_index("Feature")["Importance"])
    elif "Random Forest" in models:
        fi = pd.DataFrame(
            {"Feature": feature_columns, "Importance": models["Random Forest"].feature_importances_}
        ).sort_values("Importance", ascending=False)
        st.subheader("Feature importance (Random Forest)")
        st.bar_chart(fi.set_index("Feature")["Importance"])

# --------------------------------------------------------------------------------------
# Data overview tab
# --------------------------------------------------------------------------------------
with tab_data:
    st.subheader("Raw data preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.write(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")

    if "Revenue" in raw_df.columns:
        st.subheader("Revenue distribution")
        st.bar_chart(raw_df["Revenue"])

    if TARGET_COL in raw_df.columns:
        st.subheader("Revenue class balance")
        st.bar_chart(raw_df[TARGET_COL].value_counts())
