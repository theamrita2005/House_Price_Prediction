import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from currency import CURRENCIES, convert_from_usd, format_currency

def predict_usa_house(income, house_age, rooms, bedrooms, population):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "usa_house_price_model.joblib")
    artifacts = joblib.load(model_path)
    
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    rooms_per_bedroom = rooms / max(bedrooms, 0.1)

    input_df = pd.DataFrame([{
        'Avg. Area Income': income,
        'Avg. Area House Age': house_age,
        'Avg. Area Number of Rooms': rooms,
        'Avg. Area Number of Bedrooms': bedrooms,
        'Area Population': population,
        'Rooms_Per_Bedroom': rooms_per_bedroom
    }], columns=feature_cols)

    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)
    price_usd = model.predict(input_scaled_df)[0]
    return price_usd

def predict_india_house(sqft, bhk, bath, balcony=1.0, location="Whitefield"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "india_house_price_model.joblib")
    artifacts = joblib.load(model_path)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    # Create input feature dict matching trained columns
    input_dict = {col: 0.0 for col in feature_cols}
    input_dict['sqft'] = sqft
    input_dict['bhk'] = bhk
    input_dict['bath'] = bath
    input_dict['balcony'] = balcony

    if location in input_dict:
        input_dict[location] = 1.0

    input_df = pd.DataFrame([input_dict], columns=feature_cols)
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols)
    
    price_usd = model.predict(input_scaled_df)[0]
    return price_usd

def main():
    parser = argparse.ArgumentParser(description="House Price Predictor CLI")
    parser.add_argument("--dataset", type=str, default="usa", choices=["usa", "india"], help="Dataset (usa or india)")
    parser.add_argument("--income", type=float, default=65000, help="USA: Avg Area Income ($)")
    parser.add_argument("--age", type=float, default=6.0, help="USA: Avg House Age (years)")
    parser.add_argument("--rooms", type=float, default=7.0, help="USA: Total Rooms")
    parser.add_argument("--bedrooms", type=float, default=4.0, help="USA: Bedrooms")
    parser.add_argument("--population", type=float, default=30000, help="USA: Area Population")
    
    parser.add_argument("--sqft", type=float, default=1200, help="India: Total Sqft")
    parser.add_argument("--bhk", type=float, default=2, help="India: BHK / Bedrooms")
    parser.add_argument("--bath", type=float, default=2, help="India: Bathrooms")
    parser.add_argument("--location", type=str, default="Whitefield", help="India: Location")
    
    parser.add_argument("--currency", type=str, default="USD", choices=list(CURRENCIES.keys()), help="Currency Code")

    args = parser.parse_args()
    curr_code = args.currency.upper()

    if args.dataset.lower() == "india":
        if args.currency == "USD":
            curr_code = "INR"  # Default Indian prediction display in INR
        price_usd = predict_india_house(args.sqft, args.bhk, args.bath, 1.0, args.location)
        formatted_price = format_currency(price_usd, curr_code)
        
        print("=" * 60)
        print("        INDIAN HOUSE PRICE PREDICTION RESULTS           ")
        print("=" * 60)
        print(f"  Selected Location        : {args.location}")
        print(f"  Total Area               : {args.sqft:,.0f} sqft")
        print(f"  Bedrooms (BHK)           : {args.bhk}")
        print(f"  Bathrooms                : {args.bath}")
        print(f"  Selected Currency        : {curr_code}")
        print("-" * 60)
        print(f"  ESTIMATED HOUSE PRICE    : {formatted_price}")
        print("=" * 60)
    else:
        price_usd = predict_usa_house(args.income, args.age, args.rooms, args.bedrooms, args.population)
        formatted_price = format_currency(price_usd, curr_code)
        formatted_income = format_currency(args.income, curr_code)
        
        print("=" * 60)
        print("          USA HOUSE PRICE PREDICTION RESULTS            ")
        print("=" * 60)
        print(f"  Selected Currency        : {curr_code}")
        print(f"  Input Area Income        : {formatted_income}")
        print(f"  Input House Age          : {args.age:.1f} years")
        print(f"  Input Total Rooms        : {args.rooms:.1f}")
        print(f"  Input Bedrooms           : {args.bedrooms:.1f}")
        print(f"  Input Area Population    : {args.population:,.0f}")
        print("-" * 60)
        print(f"  ESTIMATED HOUSE PRICE    : {formatted_price}")
        print("=" * 60)

if __name__ == "__main__":
    main()
