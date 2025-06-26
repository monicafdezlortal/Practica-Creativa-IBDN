package es.upm.dit.ging.predictor

import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.{Bucketizer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.functions.{concat, from_json, lit}
import org.apache.spark.sql.types.{DataTypes, StructType}

import org.apache.spark.sql.{SparkSession, DataFrame}


object MakePrediction {

  def main(args: Array[String]): Unit = {
    println("Flight predictor starting...")

    val spark = SparkSession
      .builder
      .appName("FlightDelayPrediction")
      .getOrCreate()

    import spark.implicits._

    val base_path = "/app"

    val arrivalBucketizer = Bucketizer.load(s"$base_path/models/arrival_bucketizer_2.0.bin")
    val stringColumns = Seq("Carrier", "Origin", "Dest", "Route")

    val indexers = stringColumns.map { col =>
      col -> StringIndexerModel.load(s"$base_path/models/string_indexer_model_$col.bin")
    }.toMap

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array(
        "DepDelay", "DayOfMonth", "DayOfWeek", "DayOfYear",
        "Carrier_index", "Origin_index", "Dest_index", "Route_index"
      ))

      .setOutputCol("features")
      .setHandleInvalid("keep")

    
    val rfc = RandomForestClassificationModel.load(
      s"$base_path/models/spark_random_forest_classifier.flight_delays.5.0.bin"
    )

    // Cargar distancias desde Cassandra
    val cassandraDistances = spark.read
      .format("org.apache.spark.sql.cassandra")
      .options(Map("keyspace" -> "flights", "table" -> "origin_dest_distances"))
      .load()
   

    // Kafka input
    val kafkaDf = spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "kafka:9092")
      .option("subscribe", "flight-delay-ml-request")
      .option("startingOffsets", "earliest")
      .load()

    val flightSchema = new StructType()
      .add("Origin", DataTypes.StringType)
      .add("FlightNum", DataTypes.StringType)
      .add("DayOfWeek", DataTypes.IntegerType)
      .add("DayOfYear", DataTypes.IntegerType)
      .add("DayOfMonth", DataTypes.IntegerType)
      .add("Dest", DataTypes.StringType)
      .add("DepDelay", DataTypes.DoubleType)
      .add("Prediction", DataTypes.StringType)
      .add("Timestamp", DataTypes.TimestampType)
      .add("FlightDate", DataTypes.DateType)
      .add("Carrier", DataTypes.StringType)
      .add("UUID", DataTypes.StringType)

    val parsedDf = kafkaDf
      .selectExpr("CAST(value AS STRING)")
      .select(from_json($"value", flightSchema).as("flight"))
      .select("flight.*")

    val withRoute = parsedDf.withColumn("Route", concat($"Origin", lit("-"), $"Dest"))
    val withDistance = withRoute
      .join(cassandraDistances, Seq("Origin", "Dest"), "left")
      .withColumnRenamed("distance", "Distance")

    val indexed = indexers.foldLeft(withDistance) { case (df, (col, indexer)) =>
        indexer.transform(df)      }

        // Generación del vector de características
    val features = vectorAssembler.transform(indexed)

    // Si el modelo fue entrenado con la columna "Features_vec", renómbrala
    val renamedFeatures = features
      .withColumnRenamed("features", "Features_vec")

    // Predecir con el modelo
    rfc.setPredictionCol("PredictedDelay")
    val predictions = rfc.transform(renamedFeatures)



    // MongoDB Output
    predictions.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        try {
          println(s"MongoDB Batch $batchId con ${batchDF.count()} registros")
          val cleaned = batchDF.select(
              "Origin", "Dest", "DayOfWeek", "DayOfYear", "DayOfMonth",
              "DepDelay", "Timestamp", "Carrier", "UUID", "Route",
              "Distance", "PredictedDelay"
            )
          cleaned.write
            .format("mongodb")
            .option("spark.mongodb.connection.uri", "mongodb://mongo:27017")
            .option("spark.mongodb.database", "agile_data_science")
            .option("spark.mongodb.collection", "flight_delay_ml_response")
            .mode("append")
            .save()
        } catch {
          case e: Exception =>  
            println(s"Error en escritura a MongoDB en batch $batchId: ${e.getMessage}")
        }
      }
      .option("checkpointLocation", "/tmp/mongo-checkpoint")  // OPCIONAL pero recomendable
      .outputMode("append")
      .start() 


    // HDFS Output
    predictions.writeStream
      .format("parquet")
      .option("path", "hdfs://namenode:9000/user/spark/prediction")
      .option("checkpointLocation", "hdfs://namenode:9000/user/spark/checkpoints")
      .outputMode("append")
      .start()

    // Kafka Output
    predictions.selectExpr("to_json(struct(*)) AS value")
      .writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "kafka:9092")
      .option("topic", "flight-delay-ml-response")
      .option("checkpointLocation", "/tmp/kafka-checkpoint")
      .outputMode("append")
      .start()

    // Consola
    predictions.writeStream
      .outputMode("append")
      .format("console")
      .start()

    spark.streams.awaitAnyTermination()
  }
}
