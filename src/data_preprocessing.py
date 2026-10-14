import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_state(address):
    """Extract 2-letter state code from address string if present."""
    match = re.search(r'\b([A-Z]{2})\s+\d{5}', str(address))
    if match:
        return match.group(1)
    return "UNKNOWN"

def load_and_preprocess_usa(csv_path, test_size=0.2, random_state=42):
    """Loads and preprocesses the Kaggle USA Housing dataset."""
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    df['Rooms_Per_Bedroom'] = df['Avg. Area Number of Rooms'] / np.maximum(df['Avg. Area Number of Bedrooms'], 0.1)
    df['State'] = df['Address'].apply(extract_state)

    y = df['Price']
    feature_cols = [
        'Avg. Area Income',
        'Avg. Area House Age',
        'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms',
        'Area Population',
        'Rooms_Per_Bedroom'
    ]

    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler, feature_cols, df

def load_and_preprocess_india(csv_path, test_size=0.2, random_state=42):
    """Loads and preprocesses the Kaggle Bengaluru Indian Housing dataset."""
    df = pd.read_csv(csv_path)
    
    # Parse BHK number
    df['bhk'] = df['size'].apply(
        lambda x: float(str(x).split()[0]) if pd.notnull(x) and str(x).split()[0].isdigit() else np.nan
    )
    
    # Parse Sqft
    def convert_sqft(val):
        try:
            if '-' in str(val):
                parts = str(val).split('-')
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            return float(val)
        except:
            return np.nan

    df['sqft'] = df['total_sqft'].apply(convert_sqft)

    df = df.dropna(subset=['price', 'sqft', 'bhk', 'location'])
    df['location'] = df['location'].apply(lambda x: str(x).strip())
    
    # Consolidate rare locations
    location_stats = df['location'].value_counts()
    df['location'] = df['location'].apply(lambda x: 'other' if location_stats.get(x, 0) <= 10 else x)

    # Filter outliers
    df = df[(df['sqft'] / df['bhk']) >= 300]
    df['bath'] = df['bath'].fillna(df['bhk'])
    df['balcony'] = df['balcony'].fillna(1.0)
    df = df[df['bath'] <= df['bhk'] + 2]

    # Remove price per sqft outliers per location
    def remove_pps_outliers(df_in):
        df_out = pd.DataFrame()
        df_in['price_per_sqft'] = (df_in['price'] * 100000) / df_in['sqft']
        for key, subdf in df_in.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df_clean = remove_pps_outliers(df)

    # Target: Price in INR Rupees (1 Lakh = 100,000 INR)
    # Convert to USD equivalent for standard model comparison (1 USD ~ 83.5 INR)
    y_inr = df_clean['price'] * 100000
    y_usd = y_inr / 83.50

    # Categorical location encoding using dummy variables
    df_dummies = pd.get_dummies(df_clean['location'], drop_first=True)
    
    base_features = ['sqft', 'bhk', 'bath', 'balcony']
    X = pd.concat([df_clean[base_features], df_dummies], axis=1)
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_usd, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler, feature_cols, df_clean

def load_and_preprocess_data(csv_path, dataset_type="usa", test_size=0.2, random_state=42):
    """Generic wrapper loading USA or Indian dataset."""
    if "bengaluru" in csv_path.lower() or dataset_type.lower() == "india":
        return load_and_preprocess_india(csv_path, test_size, random_state)
    return load_and_preprocess_usa(csv_path, test_size, random_state)

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    usa_path = os.path.join(base_dir, "data", "USA_Housing.csv")
    india_path = os.path.join(base_dir, "data", "Bengaluru_House_Data.csv")

    X_tr_u, X_te_u, _, _, _, cols_u, _ = load_and_preprocess_data(usa_path, "usa")
    print(f"USA Dataset Processed: {X_tr_u.shape[0]} train rows, {len(cols_u)} features")

    X_tr_i, X_te_i, _, _, _, cols_i, _ = load_and_preprocess_data(india_path, "india")
    print(f"Indian Dataset Processed: {X_tr_i.shape[0]} train rows, {len(cols_i)} features")
