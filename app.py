import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from download_data import download_dataset
from data_preprocessing import load_and_preprocess_data
from predict import predict_usa_house, predict_india_house
from currency import CURRENCIES, convert_from_usd, format_currency
import joblib

st.set_page_config(
    page_title="Multi-Dataset House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction System")
st.markdown("""
This interactive Machine Learning application supports multiple real estate datasets from Kaggle:
- **USA Housing Dataset**
- **Indian Housing Dataset (Bengaluru, India)**

Features **Linear/Ridge Regression models**, dynamic **Multi-Currency conversion**, and interactive analytics!
""")

base_dir = os.path.dirname(os.path.abspath(__file__))
usa_csv = os.path.join(base_dir, "data", "USA_Housing.csv")
india_csv = os.path.join(base_dir, "data", "Bengaluru_House_Data.csv")

# Ensure dataset files exist
if not os.path.exists(usa_csv) or not os.path.exists(india_csv):
    download_dataset()

# Ensure model files exist
usa_model_path = os.path.join(base_dir, "models", "usa_house_price_model.joblib")
india_model_path = os.path.join(base_dir, "models", "india_house_price_model.joblib")

if not os.path.exists(usa_model_path) or not os.path.exists(india_model_path):
    from model_training import train_all_models
    train_all_models()

# Sidebar Setup
st.sidebar.header("⚙️ Settings")

dataset_choice = st.sidebar.selectbox(
    "📊 Select Dataset",
    ["USA Housing Dataset (Kaggle)", "Indian Housing Dataset - Bengaluru (Kaggle)"]
)

is_india = "Indian" in dataset_choice
current_dataset_type = "india" if is_india else "usa"
current_csv_path = india_csv if is_india else usa_csv
current_model_path = india_model_path if is_india else usa_model_path

# International Currency Selector
currency_options = list(CURRENCIES.keys())
default_curr_idx = currency_options.index("INR") if is_india else currency_options.index("USD")

selected_curr_index = st.sidebar.selectbox(
    "🌍 Select Currency",
    range(len(currency_options)),
    index=default_curr_idx,
    format_func=lambda i: CURRENCIES[currency_options[i]]["name"]
)
selected_currency = currency_options[selected_curr_index]
curr_info = CURRENCIES[selected_currency]
curr_symbol = curr_info["symbol"]
curr_rate = curr_info["rate"]

page = st.sidebar.radio("Select View", ["Price Predictor 💡", "Model Analytics 📊", "Dataset Exploration 📁"])

# Load dataset artifacts
artifacts = joblib.load(current_model_path)
metrics = artifacts["metrics"]
model = artifacts["model"]
feature_cols = artifacts["feature_cols"]

# Load preprocessed data
X_train, X_test, y_train, y_test, scaler, _, df = load_and_preprocess_data(current_csv_path, current_dataset_type)

if page == "Price Predictor 💡":
    st.header(f"💡 Predict House Price ({dataset_choice})")
    
    if is_india:
        locations_list = sorted(list(df['location'].unique()))
        col1, col2 = st.columns(2)
        
        with col1:
            location_sel = st.selectbox("Select Location in Bengaluru", locations_list)
            sqft_input = st.number_input("Total Area (sqft)", min_value=300, max_value=10000, value=1200, step=50)

        with col2:
            bhk_input = st.slider("Number of Bedrooms (BHK)", min_value=1, max_value=6, value=2, step=1)
            bath_input = st.slider("Number of Bathrooms", min_value=1, max_value=6, value=2, step=1)
            balcony_input = st.slider("Number of Balconies", min_value=0, max_value=3, value=1, step=1)

        if st.button("Calculate Indian House Price 🚀", type="primary"):
            price_usd = predict_india_house(sqft_input, bhk_input, bath_input, balcony_input, location_sel)
            formatted_price = format_currency(price_usd, selected_currency)
            
            st.success(f"### Estimated Property Price: **{formatted_price}**")
            st.info(f"""
            **Property Summary:**
            - **Location:** {location_sel}, Bengaluru
            - **Area:** {sqft_input:,} sqft
            - **Configuration:** {bhk_input} BHK, {bath_input} Bathrooms, {balcony_input} Balconies
            - **Equivalent in USD:** ${price_usd:,.2f}
            """)

    else:
        col1, col2 = st.columns(2)
        default_income_usd = 68000
        default_income_local = int(default_income_usd * curr_rate)

        with col1:
            income_input = st.slider(
                f"Average Area Income ({curr_symbol})",
                min_value=int(20000 * curr_rate),
                max_value=int(150000 * curr_rate),
                value=default_income_local,
                step=int(1000 * curr_rate) if curr_rate >= 1 else 100
            )
            income_usd = income_input / curr_rate
            house_age = st.slider("Average House Age (Years)", min_value=1.0, max_value=12.0, value=6.0, step=0.1)
            rooms = st.slider("Average Number of Rooms", min_value=2.0, max_value=12.0, value=7.0, step=0.1)

        with col2:
            bedrooms = st.slider("Average Number of Bedrooms", min_value=1.0, max_value=8.0, value=4.0, step=0.1)
            population = st.slider("Area Population", min_value=5000, max_value=80000, value=35000, step=1000)

        if st.button("Calculate USA House Price 🚀", type="primary"):
            predicted_price_usd = predict_usa_house(income_usd, house_age, rooms, bedrooms, population)
            formatted_price = format_currency(predicted_price_usd, selected_currency)
            formatted_income = format_currency(income_usd, selected_currency)
            
            st.success(f"### Estimated Property Price: **{formatted_price}**")
            st.info(f"""
            **Prediction Summary ({selected_currency}):**
            - **Area Income:** {formatted_income}
            - **House Age:** {house_age:.1f} years
            - **Rooms / Bedrooms:** {rooms:.1f} total rooms / {bedrooms:.1f} bedrooms
            - **Area Population:** {population:,.0f} residents
            """)

elif page == "Model Analytics 📊":
    st.header(f"📊 Regression Model Evaluation ({dataset_choice})")

    rmse_converted = format_currency(metrics['RMSE'], selected_currency)
    mae_converted = format_currency(metrics['MAE'], selected_currency)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score (Accuracy)", f"{metrics['R2']:.4f}")
    m2.metric("Adjusted R² Score", f"{metrics['Adj_R2']:.4f}")
    m3.metric("RMSE Error", rmse_converted)
    m4.metric("MAE Error", mae_converted)

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    y_pred_test_usd = model.predict(X_test)
    y_test_curr = y_test * curr_rate
    y_pred_curr = y_pred_test_usd * curr_rate

    scale_factor = 1e6 if curr_rate < 50 else 1e8

    with col1:
        st.subheader("Actual vs Predicted House Prices")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test_curr / scale_factor, y_pred_curr / scale_factor, alpha=0.5, color="#1f77b4")
        ax.plot([y_test_curr.min() / scale_factor, y_test_curr.max() / scale_factor], 
                [y_test_curr.min() / scale_factor, y_test_curr.max() / scale_factor], 'r--', lw=2)
        ax.set_xlabel(f"Actual Price ({curr_symbol})")
        ax.set_ylabel(f"Predicted Price ({curr_symbol})")
        ax.set_title("Actual vs Predicted Prices")
        st.pyplot(fig)

    with col2:
        st.subheader("Residual Analysis (Errors)")
        fig, ax = plt.subplots(figsize=(6, 4))
        residuals_curr = y_test_curr - y_pred_curr
        ax.hist(residuals_curr / (scale_factor / 1e3), bins=30, color="#2ca02c", edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"Residual Error ({curr_symbol})")
        ax.set_ylabel("Frequency")
        ax.set_title("Residual Error Distribution")
        st.pyplot(fig)

elif page == "Dataset Exploration 📁":
    st.header(f"📁 Kaggle Dataset Explorer ({dataset_choice})")
    st.write(f"Total Dataset Records: **{len(df):,} rows**, **{df.shape[1]} columns**")
    
    display_df = df.copy()
    if is_india:
        display_df["Price (Lakhs INR)"] = display_df["price"]
        st.dataframe(display_df[['location', 'size', 'total_sqft', 'bath', 'balcony', 'price']].head(15), use_container_width=True)
    else:
        st.dataframe(display_df.head(15), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(display_df.describe().T, use_container_width=True)
