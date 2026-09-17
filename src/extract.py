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
    
    print("Loading solutions (labels)...")
    labels_file = os.path.join(data_dir, "TRAIN_NEW", "TRAINING_SOLUTIONS.xlsx")
    labels_df = pd.read_excel(labels_file)
    
    # Merge on participant_id to ensure exact alignment
    merged_df = features_df.merge(labels_df, on="participant_id", how="inner")
    
    # Extract IDs, Features, and Target (ADHD Outcome)
    ids = merged_df["participant_id"].values
    
    # Target variable
    y = merged_df["ADHD_Outcome"].values
    
    # Everything else except ID and solution columns are features
    feature_cols = [c for c in merged_df.columns if c not in ["participant_id", "ADHD_Outcome", "Sex_F"]]
    X = merged_df[feature_cols].values
    
    return X, y, ids
