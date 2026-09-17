variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "Name of the GCS bucket for raw fMRI data"
  type        = string
}

variable "bq_dataset_id" {
  description = "BigQuery dataset ID for metadata"
  type        = string
  default     = "fmri_metadata"
}
