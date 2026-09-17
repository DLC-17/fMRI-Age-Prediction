import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.cloud_ingest import upload_to_gcs, load_metadata_to_bq


@patch("src.cloud_ingest.storage")
def test_upload_to_gcs(mock_storage):
    mock_client = MagicMock()
    mock_storage.Client.return_value = mock_client
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    result = upload_to_gcs("test-bucket", "test-file.csv", "test-blob.csv")

    mock_storage.Client.assert_called_once()
    mock_client.bucket.assert_called_with("test-bucket")
    mock_bucket.blob.assert_called_with("test-blob.csv")
    mock_blob.upload_from_filename.assert_called_with("test-file.csv")
    assert result == "gs://test-bucket/test-blob.csv"


@patch("src.cloud_ingest.bigquery")
def test_load_metadata_to_bq(mock_bigquery):
    mock_client = MagicMock()
    mock_client.project = "test-project"
    mock_bigquery.Client.return_value = mock_client
    mock_job = MagicMock()
    mock_job.output_rows = 2
    mock_client.load_table_from_dataframe.return_value = mock_job

    test_df = pd.DataFrame(
        {"subject_id": ["sub-001", "sub-002"], "age": [10.0, 15.0]}
    )

    load_metadata_to_bq("test_dataset", "test_table", test_df)

    mock_bigquery.Client.assert_called_once()
    mock_client.load_table_from_dataframe.assert_called_once()
    mock_job.result.assert_called_once()
