resource "google_storage_bucket" "fmri_data" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}

resource "google_bigquery_dataset" "fmri_dataset" {
  dataset_id                  = var.bq_dataset_id
  location                    = var.region
  description                 = "Dataset for fMRI metadata"
  
  delete_contents_on_destroy  = false
}

resource "google_bigquery_table" "metadata_table" {
  dataset_id = google_bigquery_dataset.fmri_dataset.dataset_id
  table_id   = "subject_metadata"

  schema = <<EOF
[
  {
    "name": "subject_id",
    "type": "STRING",
    "mode": "REQUIRED",
    "description": "Unique identifier for the subject"
  },
  {
    "name": "age",
    "type": "FLOAT",
    "mode": "NULLABLE",
    "description": "Age of the subject"
  },
  {
    "name": "sex",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Biological sex of the subject"
  },
  {
    "name": "scan_date",
    "type": "DATE",
    "mode": "NULLABLE",
    "description": "Date of the fMRI scan"
  },
  {
    "name": "gcs_path",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "Path to the raw TSV data in GCS"
  }
]
EOF
}
