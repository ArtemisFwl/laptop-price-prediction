import pandas as pd
import os

from src.data_cleaning import clean_data

# Resolve project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct path to raw data using os module
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "laptop_price.csv")
# Load raw data
df = pd.read_csv(DATA_PATH)


df_clean=clean_data(df)

from src.feature_engineering import engineer_features

df_clean = clean_data(df)
df_fe = engineer_features(df_clean)

print(df_fe.shape)


print(df_clean.shape)