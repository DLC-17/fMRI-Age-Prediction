import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.cloud_ingest import upload_to_gcs, load_to_bigquery, extract_from_bigquery

@patch('src.cloud_ingest.storage')
def test_upload_to_gcs(mock_storage):
    mock_client = MagicMock()
    mock_storage.Client.return_value = mock_client
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    upload_to_gcs('test-bucket', 'test-file.csv', 'test-blob.csv')
    
    mock_storage.Client.assert_called_once()
    mock_client.bucket.assert_called_with('test-bucket')
    mock_bucket.blob.assert_called_with('test-blob.csv')
    mock_blob.upload_from_filename.assert_called_with('test-file.csv')

@patch('src.cloud_ingest.bigquery')
def test_load_to_bigquery(mock_bigquery):
    mock_client = MagicMock()
    mock_bigquery.Client.return_value = mock_client
    mock_job = MagicMock()
    mock_client.load_table_from_uri.return_value = mock_job

    load_to_bigquery('gs://test-bucket/test-blob.csv', 'test_dataset.test_table')
    
    mock_bigquery.Client.assert_called_once()
    mock_client.load_table_from_uri.assert_called_once()
    mock_job.result.assert_called_once()

@patch('src.cloud_ingest.bigquery')
def test_extract_from_bigquery(mock_bigquery):
    mock_client = MagicMock()
    mock_bigquery.Client.return_value = mock_client
    mock_query_job = MagicMock()
    mock_client.query.return_value = mock_query_job
    mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mock_query_job.to_dataframe.return_value = mock_df

    df = extract_from_bigquery('SELECT * FROM test_dataset.test_table')
    
    mock_bigquery.Client.assert_called_once()
    mock_client.query.assert_called_with('SELECT * FROM test_dataset.test_table')
    pd.testing.assert_frame_equal(df, mock_df)
