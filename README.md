# 🍽️ Restaurant Revenue Class Predictor

A Streamlit web app that predicts whether a restaurant falls into the **High** or **Low** revenue class, based on features like location, cuisine, ratings, marketing budget, and reservations.

Built on top of the analysis in [`Restaurant_Revenue_prediction.ipynb`](./Restaurant_Revenue_prediction.ipynb), this app loads the trained model artifacts the notebook exports and serves live predictions.
DEMO:https://restaurantrevenueprediction-vna6tgyszpmalvxrpsklvn.streamlit.app/

## ✨ Features

- **🔮 Single prediction** — fill in one restaurant's details and get an instant High/Low revenue prediction with class probabilities
- **📄 Batch prediction** — upload a CSV of many restaurants and download predictions for all of them
- **🌲 Feature importance** — see which features drive revenue class the most (from the trained model)
- **⚡ Fast startup** — loads pre-trained artifacts instead of retraining on every run

## 🧠 How it works

The notebook's final cells train a Decision Tree classifier and export three artifacts:

| File | What it is |
|---|---|
| `restaurant_model.pkl` | Trained Decision Tree classifier |
| `scaler.pkl` | `StandardScaler` fit on the training features |
| `encoders.pkl` | `dict` of `{column_name: LabelEncoder}` for the categorical columns |

The app loads all three, then for each prediction:

1. Encodes categorical inputs (`Location`, `Cuisine`, `Parking Availability`) with the saved `LabelEncoder`s
2. Applies the saved `StandardScaler` **only if** the loaded model is a type that needs it (Logistic Regression, KNN, or SVM) — tree-based models (Decision Tree, Random Forest) are predicted on raw features, matching how they were trained in the notebook
3. Runs `model.predict()` / `model.predict_proba()` and displays the result

This means you can swap in any of the other models trained in the notebook (Logistic Regression, Random Forest, KNN, SVM) just by re-pickling it as `restaurant_model.pkl` — the app detects the model type automatically and scales inputs accordingly.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- The three exported artifacts: `restaurant_model.pkl`, `scaler.pkl`, `encoders.pkl`

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

### Add the model artifacts

Copy `restaurant_model.pkl`, `scaler.pkl`, and `encoders.pkl` (downloaded from the notebook's last cell) into the same folder as `app.py`. If they're not found, the app's sidebar will let you upload them manually at runtime instead.

### Run locally

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## 📁 Expected Feature Columns

The app expects these 15 columns, in this order (matching the notebook's training data):

| Column | Type |
|---|---|
| Location | categorical |
| Cuisine | categorical |
| Rating | numeric |
| Seating Capacity | numeric |
| Average Meal Price | numeric |
| Marketing Budget | numeric |
| Social Media Followers | numeric |
| Chef Experience Years | numeric |
| Number of Reviews | numeric |
| Avg Review Length | numeric |
| Ambience Score | numeric |
| Service Quality Score | numeric |
| Parking Availability | categorical |
| Weekend Reservations | numeric |
| Weekday Reservations | numeric |

`Name`, `Revenue`, and `Revenue_Class` are not model inputs — for batch predictions, extra columns like these in an uploaded CSV are simply ignored.

## 🗂️ Project Structure

```
.
├── app.py                              # Streamlit app
├── requirements.txt                    # Python dependencies
├── restaurant_model.pkl                # Trained model (add this yourself)
├── scaler.pkl                          # Fitted StandardScaler (add this yourself)
├── encoders.pkl                        # Fitted LabelEncoders (add this yourself)
├── Restaurant_Revenue_prediction.ipynb # Original analysis/model notebook
└── README.md
```

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — model training & inference
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## ☁️ Deploying

The easiest option is [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub, **including** `restaurant_model.pkl`, `scaler.pkl`, and `encoders.pkl`
2. Go to share.streamlit.io and connect the repo
3. Set the main file path to `app.py`
4. Deploy

> If your `.pkl` files are large, consider [Git LFS](https://git-lfs.com/) rather than committing them directly.
