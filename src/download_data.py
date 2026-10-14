import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

USA_DATA_FILE = os.path.join(DATA_DIR, "USA_Housing.csv")
USA_DATA_URL = "https://raw.githubusercontent.com/bcbarsness/machine-learning/master/USA_Housing.csv"

INDIA_DATA_FILE = os.path.join(DATA_DIR, "Bengaluru_House_Data.csv")
INDIA_DATA_URL = "https://raw.githubusercontent.com/tejas05in/pwskills-Data_Science_Masters/master/machine%20learning/assignments/Bengaluru_House_Data.csv"

def download_dataset():
    """Downloads USA Housing and Indian (Bengaluru) Kaggle datasets if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # USA Housing
    if not os.path.exists(USA_DATA_FILE):
        print(f"Downloading USA Housing dataset from {USA_DATA_URL}...")
        df_usa = pd.read_csv(USA_DATA_URL)
        df_usa.to_csv(USA_DATA_FILE, index=False)
        print(f"USA Dataset saved to {USA_DATA_FILE} (Shape: {df_usa.shape})")
    else:
        df_usa = pd.read_csv(USA_DATA_FILE)

    # Indian Housing
    if not os.path.exists(INDIA_DATA_FILE):
        print(f"Downloading Indian Housing dataset (Bengaluru) from {INDIA_DATA_URL}...")
        df_india = pd.read_csv(INDIA_DATA_URL)
        df_india.to_csv(INDIA_DATA_FILE, index=False)
        print(f"Indian Dataset saved to {INDIA_DATA_FILE} (Shape: {df_india.shape})")
    else:
        df_india = pd.read_csv(INDIA_DATA_FILE)

    return df_usa, df_india

if __name__ == "__main__":
    download_dataset()
