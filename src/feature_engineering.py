"""
feature_engineering.py

This module is responsible for creating new features from cleaned data.
All derived features that help improve model performance should be added here.
"""

import numpy as np
import pandas as pd 

def engineer_features(df):
    """
     create new features from the cleaned dataset 
     Planned features:
     - PPI Pixel per inch 
     - CPU Category simplification
     - GPU category simplification
     - Storage typr implementation 
     - Other derived numerical features 

    parameters- 
      df (pd.DataFrame): Cleaned input dataframe  
      returns:
    pd.DataFrame: Dataframe with engineered features
   
    """
    
    # ---- PPI calculation ----
    required_cols = {"screen_width", "screen_height", "inches"}
    if required_cols.issubset(df.columns):
        df["ppi"] = (
            (df["screen_width"]**2 + df["screen_height"]**2) ** 0.5
        ) / df["inches"]
    
    # ---- CPU simplification ----
    if "cpu_type" in df.columns:
        cpu = df["cpu_type"].str.lower()

        df["cpu_brand"] = cpu.apply(
            lambda x: "intel" if "intel" in x
            else "amd" if "amd" in x
            else "other"
        )

        df["cpu_tier"] = cpu.apply(
            lambda x: "i7" if "i7" in x
            else "i5" if "i5" in x
            else "i3" if "i3" in x
            else "other"
        )

    # ---- GPU simplification ----
    if "gpu_company" in df.columns:
        gpu = df["gpu_company"].str.lower()

        df["gpu_type_simple"] = gpu.apply(
            lambda x: "integrated" if x == "intel"
            else "dedicated"
        )
    
    # ---- Storage features ----
    if "memory" in df.columns:
        mem = df["memory"].str.lower()

        df["has_ssd"] = mem.str.contains("ssd").astype(int)
        df["has_hdd"] = mem.str.contains("hdd").astype(int)


    
    return df
