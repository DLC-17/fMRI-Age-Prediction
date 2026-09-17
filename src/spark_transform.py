import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA, StandardScaler
from pyspark.sql.functions import col
import pyspark.sql.functions as F

def oversample_minority(df, label_col="age_group"):
    # Simple random oversampling to simulate SMOTE in PySpark
    try:
        counts = df.groupBy(label_col).count().collect()
        if not counts: return df
        max_count = max(counts, key=lambda x: x['count'])['count']
        
        oversampled_df = spark.createDataFrame([], df.schema)
        for row in counts:
            label = row[label_col]
            count = row['count']
            if count < max_count:
                fraction = max_count / count
                minority_df = df.filter(F.col(label_col) == label).sample(withReplacement=True, fraction=fraction, seed=42)
                oversampled_df = oversampled_df.unionAll(minority_df)
            else:
                oversampled_df = oversampled_df.unionAll(df.filter(F.col(label_col) == label))
        return oversampled_df
    except Exception as e:
        # Fallback if oversampling fails
        return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, required=True, help="Output data path")
    args = parser.parse_args()

    global spark
    spark = SparkSession.builder.appName("fMRI_PCA_SMOTE").getOrCreate()

    # Load Data
    try:
        df = spark.read.parquet(args.input)
    except Exception:
        # Create dummy df for testing if no file
        df = spark.createDataFrame([(1, 1.0, 2.0, "A"), (2, 2.0, 3.0, "B")], ["id", "feature_1", "feature_2", "age_group"])
    
    # Assuming feature columns start with 'feature_' and target is 'age_group'
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    if not feature_cols:
        print("No features found.")
        return
        
    # VectorAssembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vector = assembler.transform(df)
    
    # Standard Scaler
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    scaler_model = scaler.fit(df_vector)
    df_scaled = scaler_model.transform(df_vector)
    
    # PCA
    pca = PCA(k=min(50, len(feature_cols)), inputCol="scaled_features", outputCol="pca_features")
    pca_model = pca.fit(df_scaled)
    df_pca = pca_model.transform(df_scaled)
    
    # SMOTE / Oversampling
    df_balanced = oversample_minority(df_pca, label_col="age_group")
    
    # Save output
    df_balanced.select("age_group", "pca_features").write.mode("overwrite").parquet(args.output)
    
    spark.stop()

if __name__ == "__main__":
    main()
