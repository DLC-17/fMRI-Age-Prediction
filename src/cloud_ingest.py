import os
import argparse
import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print(f"Uploading {source_file_path} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    print("Upload complete.")
    return f"gs://{bucket_name}/{destination_blob_name}"

def load_metadata_to_bq(dataset_id, table_id, metadata_df):
    """Loads a pandas dataframe into BigQuery."""
    bq_client = bigquery.Client()
    table_ref = f"{bq_client.project}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
    )

    print(f"Loading metadata to BigQuery table {table_ref}...")
    job = bq_client.load_table_from_dataframe(
        metadata_df, table_ref, job_config=job_config
    )
    job.result()  # Wait for the job to complete.
    print("Loaded {} rows into {}.".format(job.output_rows, table_ref))

def main():
    parser = argparse.ArgumentParser(description="Ingest fMRI data to GCS and metadata to BigQuery.")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset ID")
    parser.add_argument("--table", required=True, help="BigQuery table ID")
    parser.add_argument("--data-dir", required=True, help="Local directory containing fMRI TSV files and metadata.csv")
    
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: metadata file not found at {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path)
    
    # Optional: ensure required columns exist
    required_cols = {'subject_id', 'age', 'sex'}
    if not required_cols.issubset(metadata_df.columns):
        print(f"Error: metadata.csv must contain columns: {required_cols}")
        return

    gcs_paths = []
    
    for _, row in metadata_df.iterrows():
        subject_id = row['subject_id']
        tsv_filename = f"{subject_id}.tsv"
        tsv_path = os.path.join(args.data_dir, tsv_filename)
        
        if os.path.exists(tsv_path):
            gcs_path = upload_to_gcs(args.bucket, tsv_path, f"raw/{tsv_filename}")
            gcs_paths.append(gcs_path)
        else:
            print(f"Warning: TSV file for {subject_id} not found at {tsv_path}")
            gcs_paths.append(None)

    # Update metadata with GCS paths
    metadata_df['gcs_path'] = gcs_paths
    
    # Load into BigQuery (filter out missing scans optionally)
    valid_metadata = metadata_df.dropna(subset=['gcs_path']).copy()
    
    # Ensure correct types for BigQuery
    if 'scan_date' in valid_metadata.columns:
        valid_metadata['scan_date'] = pd.to_datetime(valid_metadata['scan_date']).dt.date

    load_metadata_to_bq(args.dataset, args.table, valid_metadata)

if __name__ == "__main__":
    main()
