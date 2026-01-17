import pandas as pd
import os

# Resolve project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct path to raw data using os module
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "laptop_price.csv")
# Load raw data
df = pd.read_csv(DATA_PATH)


from src.data_cleaning import clean_data
df_clean=clean_data(df)


from src.feature_engineering import engineer_features
df_fe = engineer_features(df_clean)

from src.encoding import prepare_data
X_train, X_test, y_train, y_test = prepare_data(df_fe)


print(df_fe.shape)
print(df_clean.shape)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)