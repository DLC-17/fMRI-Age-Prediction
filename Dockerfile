FROM apache/airflow:2.6.3-python3.9

USER root
# Install OpenJDK-11 for PySpark
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
USER airflow

# Install Python dependencies for ETL jobs
RUN pip install --no-cache-dir pyspark==3.4.1 scikit-learn pandas
