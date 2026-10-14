import os
import pytest
import pandas as pd
import numpy as np
import sys

# Ensure src directory is in Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from download_data import download_dataset
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate_model
from predict import predict_usa_house, predict_india_house
from currency import convert_from_usd, convert_to_usd, format_currency

@pytest.fixture
def base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_data_download(base_dir):
    df_usa, df_india = download_dataset()
    assert isinstance(df_usa, pd.DataFrame)
    assert isinstance(df_india, pd.DataFrame)
    assert len(df_usa) > 0
    assert len(df_india) > 0
    assert "Price" in df_usa.columns
    assert "price" in df_india.columns

def test_usa_preprocessing(base_dir):
    csv_path = os.path.join(base_dir, "data", "USA_Housing.csv")
    X_train, X_test, y_train, y_test, scaler, feature_cols, raw_df = load_and_preprocess_data(csv_path, "usa")
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == len(feature_cols)

def test_india_preprocessing(base_dir):
    csv_path = os.path.join(base_dir, "data", "Bengaluru_House_Data.csv")
    X_train, X_test, y_train, y_test, scaler, feature_cols, raw_df = load_and_preprocess_data(csv_path, "india")
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert "sqft" in feature_cols
    assert "bhk" in feature_cols

def test_model_training(base_dir):
    model_u, scaler_u, cols_u, art_u = train_and_evaluate_model("usa")
    assert art_u["metrics"]["R2"] > 0.85

    model_i, scaler_i, cols_i, art_i = train_and_evaluate_model("india")
    assert art_i["metrics"]["R2"] > 0.65

def test_predict_usa_house():
    price = predict_usa_house(
        income=65000,
        house_age=6.0,
        rooms=7.0,
        bedrooms=4.0,
        population=30000
    )
    assert isinstance(price, (float, np.floating))
    assert 100000 < price < 5000000

def test_predict_india_house():
    price = predict_india_house(
        sqft=1200,
        bhk=2,
        bath=2,
        balcony=1,
        location="Whitefield"
    )
    assert isinstance(price, (float, np.floating))
    assert price > 10000  # in USD

def test_currency_conversions():
    usd_val = 100.0
    eur_val = convert_from_usd(usd_val, "EUR")
    assert eur_val == 92.0

    fmt_inr = format_currency(1000.0, "INR")
    assert "₹" in fmt_inr
