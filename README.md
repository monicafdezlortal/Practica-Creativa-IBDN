## PRÁCTICA CREATIVA

Este proyecto implementa una arquitectura completa para predecir retrasos de vuelos en tiempo real utilizando las sigueintes tecnologías: 
 **Docker, Apache Kafka, Apache Spark, NiFi, MongoDB, HDFS, MLflow y Airflow**.

---

## Arquitectura general

La arquitectura está dockerizada. Incluye:

- **Flask**: aplicación web donde el usuario introduce los datos del vuelo.
  ![Flask](img/predictions.png)
- **Kafka**: sistema de mensajería para enviar/recibir peticiones y respuestas.
- **Spark**: motor de procesamiento que recibe las peticiones, predice y guarda resultados. Interfaz del Spark Master mostrando tareas y ejecución:
  ![Spark](img/spark.png)
- **MongoDB**: almacena las predicciones en base de datos.
- **HDFS**: también almacena las predicciones como archivos parquet.
Contenido del directorio `/user/spark/prediction` donde Spark almacena los `.parquet`.
  ![HDFS](img/hdfs.png)
- **NiFi**: lee las predicciones desde Kafka y las guarda cada 10 segundos en un `.txt`.
  ![NiFi](img/nifi.png)
- **Docker Compose**: despliega toda la arquitectura.

Hemos desplegado por local atraves de la terminal:
- **MLflow**: guarda los experimentos de entrenamiento.
- **Airflow**: automatiza el entrenamiento del modelo con MLflow.
  ![Airflow](img/airflow.png)

---



## Cómo ejecutar el sistema

1. **Levantar la arquitectura**:

```bash
docker-compose up --build
```

2. **Acceder a los servicios**:

| Servicio     | URL                      |
|--------------|--------------------------|
| Flask Web App | http://localhost:5001     |
| Airflow      | http://localhost:8180     |
| NiFi         | https://localhost:8443    |
| Spark        | http://localhost:8080     |

---

## Flujo de datos

1. El usuario introduce un vuelo en la web (Flask).
2. Flask envía los datos al topic de Kafka: `flight-delay-ml-request`.
3. Spark lee esos datos, hace la predicción.
4. Spark guarda la predicción en:
   - MongoDB
   - HDFS
   - Kafka (topic: `flight-delay-ml-response`)
5. La predicción vuelve a Flask en tiempo real desde Kafka.
6. NiFi también lee las predicciones desde Kafka cada 10s y las guarda en un archivo `.txt`.

---

## Entrenamiento del modelo

- El script `train_spark_mllib_model.py` entrena un modelo Random Forest con PySpark.
- Se ejecuta desde Airflow como una tarea programada.
- MLflow registra todos los parámetros, métricas y el modelo final.


---

## Funcionalidades implementadas (puntos del examen)

- [x] Dockerizar toda la arquitectura (Flask, Spark, Kafka, Mongo, etc.)
- [x] Publicar predicciones en Kafka y Mongo
- [x] Leer predicciones con NiFi y guardarlas cada 10s en un fichero `.txt`
- [x] Guardar predicciones también en HDFS
- [x] Entrenar el modelo desde Airflow con MLflow

---

## Autores

- [Mónica Fernández Lortal]  
- [Ana Li Camello Serrano]
