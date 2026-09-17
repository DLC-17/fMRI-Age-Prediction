import argparse
import os
import sys

# Ensure the root project directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extract import load_data
from src.transform import apply_pca, apply_smote
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="WiDS ADHD Prediction Pipeline")

    parser.add_argument(
        "--stage",
        type=str,
        choices=["extract", "transform", "train", "all"],
        required=True,
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/wids2025/", help="Directory containing the WiDS dataset."
    )

    args = parser.parse_args()

    if args.stage in ["extract", "all"]:
        print("Running extraction...")
        X, y, ids = load_data(args.data_dir)
        print(f"Extraction complete. Loaded {len(ids)} subjects with {X.shape[1]} features each.")
        print(f"Target Age Range: {min(y)} to {max(y)} years")

    if args.stage in ["transform", "all"]:
        print("Running transformation...")
        if args.stage != "all":
            X, y, ids = load_data(args.data_dir)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # PCA Dimensionality reduction
        X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=40)
        
        print(f"Transformation complete. Reduced to {X_train_pca.shape[1]} PCA components.")
        print(f"Training set has {len(y_train)} samples.")

    if args.stage in ["train", "all"]:
        print("Running training...")
        if args.stage != "all":
            print("Please run with --stage all to pass data through the pipeline.")
            return
            
        from src.train import train_xgboost, evaluate_model

        model = train_xgboost(X_train_pca, y_train)
        rmse, r2 = evaluate_model(model, X_test_pca, y_test)
        
        print(f"\n--- Final Results ---")
        print(f"RMSE: {rmse:.4f} years")
        print(f"R²:   {r2:.4f}")
        
        # Save model for the API
        model.save_model("xgboost_fmri.json")
        print("Saved model to xgboost_fmri.json")

if __name__ == "__main__":
    main()
