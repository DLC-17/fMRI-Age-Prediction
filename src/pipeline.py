import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="fMRI Age Prediction Pipeline")
    
    parser.add_argument('--stage', type=str, choices=['extract', 'transform', 'train', 'all'],
                        required=True, help='Pipeline stage to execute.')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Directory containing the data.')
    
    args = parser.parse_args()
    
    if args.stage in ['extract', 'all']:
        print("Running extraction...")
        # from extract import load_data
        # data, ids = load_data(args.data_dir)
        print("Extraction complete.")
        
    if args.stage in ['transform', 'all']:
        print("Running transformation...")
        # from transform import apply_pca, apply_smote
        print("Transformation complete.")
        
    if args.stage in ['train', 'all']:
        print("Running training...")
        # from train import train_random_forest, train_xgboost
        print("Training complete.")
        
if __name__ == '__main__':
    main()
