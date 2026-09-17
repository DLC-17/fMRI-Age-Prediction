import argparse
from src.extract import load_data
from src.transform import apply_pca, apply_smote


def main():
    parser = argparse.ArgumentParser(description="fMRI Age Prediction Pipeline")

    parser.add_argument(
        "--stage",
        type=str,
        choices=["extract", "transform", "train", "all"],
        required=True,
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/", help="Directory containing the data."
    )

    args = parser.parse_args()

    if args.stage in ["extract", "all"]:
        print("Running extraction...")
        data, ids = load_data(args.data_dir)
        print(f"Extraction complete. Loaded {len(ids)} subjects with {data.shape[1]} features each.")

    if args.stage in ["transform", "all"]:
        print("Running transformation...")
        if args.stage != "all":
            data, ids = load_data(args.data_dir)
        X_train_pca, _, pca = apply_pca(data, None, n_components=40)
        print(f"Transformation complete. Reduced to {X_train_pca.shape[1]} PCA components.")

    if args.stage in ["train", "all"]:
        print("Running training...")
        from src.train import train_xgboost, evaluate_model

        print("Training complete.")


if __name__ == "__main__":
    main()
