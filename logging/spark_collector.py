from pyspark.sql import SparkSession
from pyspark.sql.functions import mean
import os

spark = SparkSession.builder.appName('metric_collector').getOrCreate()
df = spark.read.json('metrics/*.json')
summary = df.groupBy('run').agg(mean('throughput').alias('throughput'), mean('loss').alias('loss'))
jdbc_url = os.environ['JDBC_URL']
summary.write.jdbc(url=jdbc_url, table='run_metrics', mode='append')
spark.stop()
