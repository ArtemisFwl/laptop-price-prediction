"""
model_evaluation.py

Evaluates regression models using standard metrics
"""

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

def evaluate_regression (model, X_test, y_test):
    y_pred= model.predict(X_test)
    mae=mean_absolute_error(y_test, y_pred)
    rmse=np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def get_feature_importance(model, feature_names):
    """
    Returns feature importance for tree based model
    """
    importance=model.feature_importances_

    return(
        pd.DataFrame({
            "feature":feature_names,
            "importance": importance
        })
        .sort_values(by="importance",ascending=False)
        .reset_index(drop=True)
    )