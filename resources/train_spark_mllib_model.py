import sys, os, re

from os import environ


# 1. Importar la librería MLflow

import mlflow

import mlflow.spark # Específico para guardar modelos Spark MLlib



def main(base_path):


  base_path = base_path if base_path else "."


  APP_NAME = "train_spark_mllib_model.py"


  # If there is no SparkSession, create the environment

  try:

    sc and spark

  except (NameError, UnboundLocalError) as e:

    import findspark

    findspark.init()

    import pyspark

    import pyspark.sql



    sc = pyspark.SparkContext()

    spark = pyspark.sql.SparkSession(sc).builder.appName(APP_NAME).getOrCreate()



  # 2. Iniciar una corrida (run) de MLflow al principio de la función main

  #    Esto crea un nuevo experimento en MLflow cada vez que se ejecuta el script.

  with mlflow.start_run() as run:

    run_id = run.info.run_id

    experiment_id = run.info.experiment_id

    print(f"MLflow Run ID: {run_id}, Experiment ID: {experiment_id}")

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")





    from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, DateType, TimestampType

    from pyspark.sql.types import StructType, StructField

    from pyspark.sql.functions import udf



    schema = StructType([

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


    input_path = "/home/monica.fernandez/data/simple_flight_delay_features.jsonl.bz2"


    # Nota: Asegúrate de que esta ruta sea accesible desde el contenedor Spark

    # Si 'data' es un volumen en tu docker-compose, asegúrate de que el path sea el correcto.

    # Por ejemplo, si el volumen es ./data:/some_path, entonces input_path debe usar /some_path

    features = spark.read.json(input_path, schema=schema)

    features.first()



    # Check for nulls in features before using Spark ML

    null_counts = [(column, features.where(features[column].isNull()).count()) for column in features.columns]

    cols_with_nulls = filter(lambda x: x[1] > 0, null_counts)

    print(list(cols_with_nulls))



    # Add a Route variable to replace FlightNum

    from pyspark.sql.functions import lit, concat

    features_with_route = features.withColumn(

      'Route',

      concat(

        features.Origin,

        lit('-'),

        features.Dest

      )

    )

    features_with_route.show(6)



    # Use pysmark.ml.feature.Bucketizer to bucketize ArrDelay into on-time, slightly late, very late (0, 1, 2)

    from pyspark.ml.feature import Bucketizer



    # Setup the Bucketizer

    splits = [-float("inf"), -15.0, 0, 30.0, float("inf")]

    arrival_bucketizer = Bucketizer(

      splits=splits,

      inputCol="ArrDelay",

      outputCol="ArrDelayBucket"

    )



    # Save the bucketizer (Optional, MLflow puede registrarlo como artefacto)

    arrival_bucketizer_path = "/models/arrival_bucketizer_2.0.bin"

    arrival_bucketizer.write().overwrite().save(arrival_bucketizer_path)



    # Apply the bucketizer

    ml_bucketized_features = arrival_bucketizer.transform(features_with_route)

    ml_bucketized_features.select("ArrDelay", "ArrDelayBucket").show()



    # Extract features tools in with pyspark.ml.feature

    from pyspark.ml.feature import StringIndexer, VectorAssembler



    # Turn category fields into indexes

    for column in ["Carrier", "Origin", "Dest", "Route"]:

      string_indexer = StringIndexer(

        inputCol=column,

        outputCol=column + "_index"

      )



      string_indexer_model = string_indexer.fit(ml_bucketized_features)

      ml_bucketized_features = string_indexer_model.transform(ml_bucketized_features)



      # Drop the original column

      ml_bucketized_features = ml_bucketized_features.drop(column)



      # Save the pipeline model (Optional, MLflow puede registrarlo como artefacto)

      string_indexer_output_path = f"/models/string_indexer_model_{column}.bin"



      string_indexer_model.write().overwrite().save(string_indexer_output_path)



    # Combine continuous, numeric fields with indexes of nominal ones

    # ...into one feature vector

    numeric_columns = [

      "DepDelay", "Distance",

      "DayOfMonth", "DayOfWeek",

      "DayOfYear"]

    index_columns = ["Carrier_index", "Origin_index",

                     "Dest_index", "Route_index"]

    vector_assembler = VectorAssembler(

      inputCols=numeric_columns + index_columns,

      outputCol="Features_vec"

    )

    final_vectorized_features = vector_assembler.transform(ml_bucketized_features)



    # Save the numeric vector assembler (Optional, MLflow puede registrarlo como artefacto)

    vector_assembler_path = "/models/numeric_vector_assembler.bin"

    vector_assembler.write().overwrite().save(vector_assembler_path)



    # Drop the index columns

    for column in index_columns:

      final_vectorized_features = final_vectorized_features.drop(column)



    # Inspect the finalized features

    final_vectorized_features.show()



    # Instantiate and fit random forest classifier on all the data

    from pyspark.ml.classification import RandomForestClassifier



    # 3. Registrar los parámetros del modelo con MLflow

    # Extraemos los parámetros que vas a registrar

    num_trees = 10 # Tu valor actual

    max_bins = 4657 # Tu valor actual

    max_memory_in_mb = 1024 # Tu valor actual



    mlflow.log_param("features_col", "Features_vec")

    mlflow.log_param("label_col", "ArrDelayBucket")

    mlflow.log_param("prediction_col", "Prediction")

    mlflow.log_param("num_trees", num_trees)

    mlflow.log_param("max_bins", max_bins)

    mlflow.log_param("max_memory_in_mb", max_memory_in_mb)



    rfc = RandomForestClassifier(

      featuresCol="Features_vec",

      labelCol="ArrDelayBucket",

      predictionCol="Prediction",

      numTrees=num_trees, # Usar la variable para el registro

      maxBins=max_bins,   # Usar la variable para el registro

      maxMemoryInMB=max_memory_in_mb

    )

    model = rfc.fit(final_vectorized_features)



    # Save the new model over the old one (Optional, MLflow lo guarda también)

    model_output_path = "/models/spark_random_forest_classifier.flight_delays.5.0.bin"

    model.write().overwrite().save(model_output_path)



    # Evaluate model using test data

    predictions = model.transform(final_vectorized_features)



    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    evaluator = MulticlassClassificationEvaluator(

      predictionCol="Prediction",

      labelCol="ArrDelayBucket",

      metricName="accuracy"

    )

    accuracy = evaluator.evaluate(predictions)

    print("Accuracy = {}".format(accuracy))



    # 4. Registrar la métrica de rendimiento con MLflow

    mlflow.log_metric("accuracy", accuracy)



    # Check the distribution of predictions

    predictions.groupBy("Prediction").count().show()



    # Check a sample

    predictions.sample(False, 0.001, 18).orderBy("CRSDepTime").show(6)



    # 5. Guardar el modelo Spark MLlib con MLflow

    # Esto guarda el modelo en el servidor de MLflow (artifact store) y lo asocia con la corrida.

    mlflow.spark.log_model(

        spark_model=model,

        artifact_path="random-forest-model", # Nombre dentro del artifact store de MLflow

        registered_model_name="FlightDelayRandomForestModel" # Nombre para registrar el modelo en el Model Registry (opcional)

    )

    print("MLflow: Model and metrics logged successfully.")



  # 6. Finalizar la sesión de Spark

  spark.stop()



if __name__ == "__main__":

  # Asegúrate de que sys.argv[1] sea el base_path correcto (e.g., /opt/airflow/dags)

  main(sys.argv[1])