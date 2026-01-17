'''
encoding.py

This module handles feature selection, categorical encoding 
and train_test_splitting to prepare data for model training 

'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(df):

    #Target and feature selection 
    target = "price_euro"
    feature_cols = [
        "inches",
        "ram_gb",
        "weight_kg",
        "ppi",
        "has_ssd",
        "has_hdd",
        "cpu_brand",
        "cpu_tier",
        "gpu_type_simple"
    ]
    
    
    y=df[target]
    X=df[feature_cols]

    #One Hot Encoding
    cat_cols=["cpu_brand", "cpu_tier", "gpu_type_simple"]
    X=pd.get_dummies(X, columns=cat_cols, drop_first=True)

    #Train test split
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)
    
    num_cols = [
    "inches",
    "ram_gb",
    "weight_kg",
    "ppi",
    "has_ssd",
    "has_hdd"
    ]

    scaler=StandardScaler()
    
    X_train[num_cols]=scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test 
