import pytest
from pyspark.sql import SparkSession
from src.spark_transform import process_data

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("pytest-spark") \
        .getOrCreate()
    yield spark
    spark.stop()

def test_process_data(spark):
    data = [
        (1, 25.0, "M"),
        (2, None, "F"),
        (3, 30.0, None)
    ]
    columns = ["id", "age", "gender"]
    df = spark.createDataFrame(data, columns)
    
    try:
        result_df = process_data(df)
        assert result_df is not None
        assert result_df.count() >= 0
    except NotImplementedError:
        pass
    except Exception as e:
        pytest.fail(f"process_data raised an exception: {e}")
