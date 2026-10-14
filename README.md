# 🏠 House Price Prediction using Linear Regression

A complete end-to-end Machine Learning project in Python that predicts house prices based on features such as income, house age, number of rooms, bedrooms, location population, and derived metrics.

## 📌 Features

- **Dataset Ingestion**: Automatically downloads and manages the Kaggle USA Housing dataset (`5,000` property records).
- **Data Preprocessing & Feature Engineering**: Handles missing values, extracts structural features (Rooms-to-Bedrooms ratio, state metadata), and applies `StandardScaler` feature scaling.
- **Model Training & Evaluation**: Trains a scikit-learn `LinearRegression` model achieving an **R² Score of 0.9180** (91.8% accuracy). Evaluates MAE, MSE, RMSE, and Adjusted R².
- **Model Persistence**: Saves model weights, scaler, and evaluation metrics using `joblib`.
- **Command-Line Inference**: Instant prediction CLI (`src/predict.py`) for custom house input parameters.
- **Interactive Web UI**: Streamlit web app (`app.py`) for real-time house price estimation, model evaluation plots, and EDA dataset exploration.
- **Automated Testing**: Unit tests (`tests/test_pipeline.py`) powered by `pytest`.

---

## 🛠️ Project Structure

```text
House_Price_Prediction/
├── app.py                     # Streamlit Interactive Web Application
├── requirements.txt           # Required Python packages
├── README.md                  # Project documentation
├── data/
│   └── USA_Housing.csv        # Kaggle USA Housing dataset
├── models/
│   └── house_price_model.joblib # Saved trained model & scaler artifacts
├── src/
│   ├── download_data.py       # Downloads Kaggle dataset
│   ├── data_preprocessing.py   # Data cleaning & feature engineering
│   ├── model_training.py      # Model training & evaluation
│   └── predict.py             # Inference script & CLI interface
└── tests/
    └── test_pipeline.py       # Pytest unit tests
```

---

## 🚀 Quick Start & Usage Guide

### 1. Installation

Ensure Python 3.10+ is installed, then install required packages:

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python src/download_data.py
```

### 3. Train the Linear Regression Model

```bash
python src/model_training.py
```

Output:
- **R² Score**: `0.9180`
- **Adjusted R²**: `0.9175`
- **RMSE**: `$100,448.49`
- **MAE**: `$80,881.07`

### 4. Run Price Predictions (CLI)

Pass house characteristics via CLI arguments:

```bash
python src/predict.py --income 75000 --age 6.5 --rooms 7.5 --bedrooms 4 --population 35000
```

### 5. Launch Interactive Web Dashboard (Streamlit)

```bash
streamlit run app.py
```

### 6. Run Automated Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

---

## 📊 Model Evaluation Metrics

| Metric | Score |
|---|---|
| **R² Score** | `0.9180` |
| **Adjusted R²** | `0.9175` |
| **Root Mean Squared Error (RMSE)** | `$100,448.49` |
| **Mean Absolute Error (MAE)** | `$80,881.07` |

---

## 📜 License

Distributed under the MIT License.
