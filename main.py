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


from src.model_training import train_linear_regression
model = train_linear_regression(X_train, y_train)

from src.model_evaluation import evaluate_regression 
metrics=evaluate_regression(model,X_test, y_test)

print("Evaluation after Scaling:", metrics)