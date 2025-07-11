import sys, os, re
from os import environ

import mlflow
import mlflow.spark

mlflow.set_tracking_uri("file:///home/monica.fernandez/practica_creativa/mlruns")

def main(base_path):
    base_path = base_path if base_path else "."
    models_dir = os.path.join(base_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    APP_NAME = "train_spark_mllib_model.py"

    try:
        sc and spark
    except (NameError, UnboundLocalError):
        import findspark
        findspark.init()
        import pyspark
        import pyspark.sql
        sc = pyspark.SparkContext()
        spark = pyspark.sql.SparkSession(sc).builder \
            .appName(APP_NAME) \
            .config("spark.cassandra.connection.host", "localhost") \
            .getOrCreate()

    mlflow.set_experiment("default")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType, TimestampType, StructType, StructField
        from pyspark.sql.functions import lit, concat

        schema = StructType([
         #   StructField("Distance", DoubleType(), True),
            StructField("ArrDelay", DoubleType(), True),
            StructField("CRSArrTime", TimestampType(), True),
            StructField("CRSDepTime", TimestampType(), True),
            StructField("Carrier", StringType(), True),
            StructField("DayOfMonth", IntegerType(), True),
            StructField("DayOfWeek", IntegerType(), True),
            StructField("DayOfYear", IntegerType(), True),
            StructField("DepDelay", DoubleType(), True),
            StructField("Dest", StringType(), True),
            StructField("FlightDate", DateType(), True),
            StructField("FlightNum", StringType(), True),
            StructField("Origin", StringType(), True),
        ])

        input_path = os.path.join(base_path, "data", "simple_flight_delay_features.jsonl.bz2")
        features = spark.read.json(input_path, schema=schema)

        # Añadir columna Route
        features = features.withColumn('Route', concat(features.Origin, lit('-'), features.Dest))

        # Renombrar columnas para coincidir con Cassandra
        features = features.withColumnRenamed("Origin", "origin").withColumnRenamed("Dest", "dest")

        # Leer distancias desde Cassandra
        distances_df = spark.read \
            .format("org.apache.spark.sql.cassandra") \
            .options(table="origin_dest_distances", keyspace="flights") \
            .load()

        # Hacer join con distancias
        features = features.join(distances_df, on=["origin", "dest"], how="left")

        # Debug opcional
        print("Columnas tras join con Cassandra:", features.columns)
        features.select("origin", "dest", "distance").show(5)

        # Verifica si 'distance' existe tras el join
        if "distance" not in features.columns:
            raise Exception("Distance column not found after join with Cassandra.")

        # Renombrar columnas de vuelta a mayúsculas
        features = features.withColumnRenamed("origin", "Origin") \
                           .withColumnRenamed("dest", "Dest") \
                           .withColumnRenamed("distance", "Distance")


        # Verifica si falta Distance
        if "Distance" not in features.columns:
            raise Exception("Distance column not found after join with Cassandra.")

        # Transformar target en categorías
        from pyspark.ml.feature import Bucketizer
        splits = [-float("inf"), -15.0, 0, 30.0, float("inf")]
        arrival_bucketizer = Bucketizer(splits=splits, inputCol="ArrDelay", outputCol="ArrDelayBucket")
        arrival_bucketizer_path = os.path.join(models_dir, "arrival_bucketizer_2.0.bin")
        arrival_bucketizer.write().overwrite().save(arrival_bucketizer_path)

        ml_bucketized_features = arrival_bucketizer.transform(features)

        # Indexar variables categóricas
        from pyspark.ml.feature import StringIndexer, VectorAssembler
        for column in ["Carrier", "Origin", "Dest", "Route"]:
            indexer = StringIndexer(inputCol=column, outputCol=column + "_index")
            model = indexer.fit(ml_bucketized_features)
            ml_bucketized_features = model.transform(ml_bucketized_features).drop(column)
            model.write().overwrite().save(os.path.join(models_dir, f"string_indexer_model_{column}.bin"))

        # Ensamblar vector de características
        numeric_columns = ["DepDelay", "Distance", "DayOfMonth", "DayOfWeek", "DayOfYear"]
        index_columns = ["Carrier_index", "Origin_index", "Dest_index", "Route_index"]
        vector_assembler = VectorAssembler(
            inputCols=numeric_columns + index_columns,
            outputCol="Features_vec"
        )
        final_vectorized_features = vector_assembler.transform(ml_bucketized_features)
        vector_assembler.write().overwrite().save(os.path.join(models_dir, "numeric_vector_assembler.bin"))

        for col in index_columns:
            final_vectorized_features = final_vectorized_features.drop(col)

        final_vectorized_features.show()

        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml import Pipeline

        rfc = RandomForestClassifier(
            featuresCol="Features_vec",
            labelCol="ArrDelayBucket",
            predictionCol="Prediction",
            numTrees=10,
            maxBins=4657,
            maxMemoryInMB=1024
        )

        pipeline = Pipeline(stages=[rfc])
        model = pipeline.fit(final_vectorized_features)

        # Guardar el modelo con el mismo nombre original
       # model.write().overwrite().save(os.path.join(models_dir, "spark_random_forest_classifier.flight_delays.5.0.bin"))
        local_model_path = os.path.join(models_dir, "spark_random_forest_classifier.flight_delays.5.0.bin")
        model.write().overwrite().save(local_model_path)


        # Evaluar
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        predictions = model.transform(final_vectorized_features)
        evaluator = MulticlassClassificationEvaluator(
            predictionCol="Prediction",
            labelCol="ArrDelayBucket",
            metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        print(f"Accuracy: {accuracy:.4f}")

        mlflow.log_param("num_trees", 10)
        mlflow.log_param("max_bins", 4657)
        mlflow.log_param("max_memory_in_mb", 1024)
        mlflow.log_metric("accuracy", accuracy)

        predictions.groupBy("Prediction").count().show()
        predictions.sample(False, 0.001, 18).orderBy("CRSDepTime").show(6)

        mlflow.spark.log_model(
            spark_model=model,
            artifact_path="random-forest-model"
            # registered_model_name="FlightDelayRandomForestModel" 
        )

        print("MLflow: Model and metrics logged successfully.")

    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
