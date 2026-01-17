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

from src.model_training import train_ridge_regression
ridge_model = train_ridge_regression(X_train, y_train, alpha=1.0)

from src.model_training import train_random_forest
rf_model=train_random_forest(X_train, y_train)
print("Random Forest trained")

from src.model_evaluation import evaluate_regression 
rf_metrics=evaluate_regression(rf_model,X_test, y_test)
print("Random Forest Evaluation:", rf_metrics)

from src.model_evaluation import get_feature_importance
feature_names=X_train.columns
fi_df=get_feature_importance(rf_model, feature_names)

print("\nTop 10 Important Features:")
print(fi_df.head(10))


from src.model_persistence import save_model
save_model(rf_model, "models/random_forest.joblib" )
print("Random Forest model saved")