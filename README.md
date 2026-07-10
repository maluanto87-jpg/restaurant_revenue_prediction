# 🍽️ Restaurant Revenue Class Predictor

A Streamlit web app that predicts whether a restaurant falls into the **High** or **Low** revenue class, based on features like location, cuisine, ratings, marketing budget, and reservations.

Built on top of the analysis in [`Restaurant_Revenue_prediction.ipynb`](./Restaurant_Revenue_prediction.ipynb), this app reproduces the notebook's full pipeline — preprocessing, encoding, scaling, and training five classification models — directly in the browser.
Demo: https://restaurantrevenueprediction-vna6tgyszpmalvxrpsklvn.streamlit.app/#result
## ✨ Features

- **📂 Upload your dataset** — bring your own `restaurant_classification.csv`
- **🔮 Live predictions** — fill in a restaurant's details and get an instant High/Low revenue prediction with class probabilities
- **🤖 Multi-model training** — trains Logistic Regression, Decision Tree, Random Forest, KNN, and SVM, then auto-selects the best performer by accuracy
- **📊 Model comparison** — accuracy, precision, recall, and F1 score for every model, plus confusion matrices
- **🌲 Feature importance** — see which features drive revenue class the most (via Random Forest)
- **🗂️ Data overview** — quick look at the raw data, revenue distribution, and class balance

## 🧠 Pipeline

The app mirrors the notebook's original workflow:

1. Load `restaurant_classification.csv`
2. Drop the `Name` column; label-encode categorical features (`Location`, `Cuisine`, `Parking Availability`)
3. Split features (`X`) from the target (`y = Revenue_Class`)
4. Train/test split (80/20, stratified)
5. Scale features with `StandardScaler` (used for Logistic Regression, KNN, and SVM)
6. Train all five models and evaluate on the held-out test set
7. Use the best-performing model for predictions in the UI

> **Note:** The original notebook only saved its `LabelEncoder`s (`encoders.pkl`) and loaded data from Google Drive — it never exported a trained model file. This app trains fresh, in-session, from whatever CSV you upload, so it always reflects the current best model rather than a stale saved one.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- The dataset: `restaurant_classification.csv` (same schema as used in the notebook)

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), and upload your `restaurant_classification.csv` in the sidebar.

## 📁 Expected Dataset Columns

| Column | Type |
|---|---|
| Name | text (dropped, not used as a feature) |
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
| Revenue | numeric (dropped, not used as a feature) |
| Revenue_Class | target (0 = Low, 1 = High) |

## 🗂️ Project Structure

```
.
├── app.py                              # Streamlit app
├── requirements.txt                    # Python dependencies
├── Restaurant_Revenue_prediction.ipynb # Original analysis/model notebook
└── README.md
```

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — model training & evaluation
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling

## ☁️ Deploying

The easiest option is [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub
2. Go to share.streamlit.io and connect the repo
3. Set the main file path to `app.py`
4. Deploy — users upload their own CSV once the app is live
