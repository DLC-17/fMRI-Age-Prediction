import pandas as pd
import numpy as np
import os

def load_data(data_dir):
    """
    Loads the functional connectome matrices and corresponding ADHD labels 
    from the WiDS Datathon 2025 dataset.
    """
    print("Loading connectome features...")
    features_file = os.path.join(data_dir, "TRAIN_NEW", "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
    features_df = pd.read_csv(features_file)
    
    print("Loading quantitative metadata (for Age)...")
    labels_file = os.path.join(data_dir, "TRAIN_NEW", "TRAIN_QUANTITATIVE_METADATA_new.xlsx")
    labels_df = pd.read_excel(labels_file)
    
    # Merge on participant_id to ensure exact alignment
    merged_df = features_df.merge(labels_df, on="participant_id", how="inner")
    
    # Drop rows where age is missing
    merged_df = merged_df.dropna(subset=["MRI_Track_Age_at_Scan"])
    
    # Extract IDs, Features, and Target (Age)
    ids = merged_df["participant_id"].values
    y = merged_df["MRI_Track_Age_at_Scan"].astype(float).values
    
    # Exclude metadata columns to isolate the connectome features
    feature_cols = [c for c in merged_df.columns if c not in labels_df.columns and c != "participant_id"]
    X = merged_df[feature_cols].values
    
    return X, y, ids
