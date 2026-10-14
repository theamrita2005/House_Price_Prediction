import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate_model(dataset_type="usa"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    if dataset_type.lower() == "india":
        csv_path = os.path.join(base_dir, "data", "Bengaluru_House_Data.csv")
        save_filename = "india_house_price_model.joblib"
    else:
        csv_path = os.path.join(base_dir, "data", "USA_Housing.csv")
        save_filename = "usa_house_price_model.joblib"

    model_save_path = os.path.join(model_dir, save_filename)
    default_save_path = os.path.join(model_dir, "house_price_model.joblib")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_cols, raw_df = load_and_preprocess_data(csv_path, dataset_type)

    # Ridge / Linear Regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predictions
    y_pred_test = model.predict(X_test)

    # Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    print("=" * 60)
    print(f"   HOUSE PRICE PREDICTION ({dataset_type.upper()}) - MODEL EVALUATION   ")
    print("=" * 60)
    print(f"Mean Absolute Error (MAE in USD):      ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE in USD): ${rmse:,.2f}")
    print(f"R² Score:                               {r2:.4f}")
    print(f"Adjusted R² Score:                      {adj_r2:.4f}")
    print("=" * 60)

    # Save artifacts
    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "dataset_type": dataset_type,
        "metrics": {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Adj_R2": adj_r2
        }
    }
    joblib.dump(artifacts, model_save_path)
    if dataset_type.lower() == "usa":
        joblib.dump(artifacts, default_save_path)

    print(f"Trained model saved to {model_save_path}")

    return model, scaler, feature_cols, artifacts

def train_all_models():
    print("--- Training USA Housing Model ---")
    train_and_evaluate_model("usa")
    print("\n--- Training Indian (Bengaluru) Housing Model ---")
    train_and_evaluate_model("india")

if __name__ == "__main__":
    train_all_models()
