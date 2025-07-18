## PRÁCTICA CREATIVA

En esta última práctica de la asignatura de Ingeniería de Big Data en la Nube, se nos pedía implementar una arquitectura completa para predecir retrasos de vuelos en tiempo real utilizando las siguientes tecnologías: 
 **Docker, Apache Kafka, Apache Spark, NiFi, MongoDB, HDFS, MLflow y Airflow, Cassandra y WebSocket.**. A continuación se describe la arquitectura para el correcto despliegue de la práctica.

---

## Arquitectura general

La arquitectura está dockerizada, es decir, cada tecnología está en un contenedor Docker. Esto nos ha permitido poder levantar toda la arquitectura de una vez automatizando el proceso.
Describimos cada una:

- **Flask**: aplicación web donde el usuario introduce los datos del vuelo.
  
  ![Flask](img/predictions.png)
  
- **Kafka**: sistema de mensajería para enviar/recibir peticiones y respuestas.
  
   ![Kafka](img/kafka.png)
  
- **Spark**: motor de procesamiento que recibe las peticiones, predice y guarda resultados. Interfaz del Spark Master mostrando tareas y ejecución:
  
  ![Spark](img/spark.png)
  
- **MongoDB**: almacena las predicciones en base de datos.
  ![mongo](img/mongo.jpeg)
  
- **HDFS**: también almacena las predicciones como archivos parquet.
  
Contenido del directorio `/user/spark/prediction` donde Spark almacena los `.parquet`.
  ![HDFS](img/hdfs.png)
  ![HDFS](img/hdfs.jpeg)

- **NiFi**: lee las predicciones desde Kafka y las guarda cada 10 segundos en un `.txt`.
  
  ![NiFi](img/nifi.png)
  
- **Docker Compose**: despliega toda la arquitectura.

Hemos desplegado por local atraves de la terminal:

- **MLflow**: guarda los experimentos de entrenamiento.
  
- **Airflow**: automatiza el entrenamiento del modelo con MLflow.
  
  ![Airflow](img/airflow.png)


- **Websockets**: permite comunicación bidireccional en tiempo real entre cliente y servidor sobre una única conexión TCP.
  
  ![Websockets](img/websockets.png)

- **Cassandra**: base de datos NoSQL distribuida diseñada para manejar grandes volúmenes de datos con alta disponibilidad y sin un único punto de falla.
  ![Cassandra](img/cassandra2.png)
  ![Cassandra](img/cassandra3.png)
  ![Cassandra](img/cassandra1.png)
  ![Cassandra](img/cassandra.png)

---



## Cómo ejecutar el sistema

1. **Levantar la arquitectura**:

```bash
docker compose up
```

2. **Acceder a los servicios**:

| Servicio     | URL                      |
|--------------|--------------------------|
| Flask Web App | http://localhost:5001    |
| HDFS         |  http://localhost:9870    |
| NiFi         | https://localhost:8443    |
| Spark        | http://localhost:8080     |
| Airflow      | http://localhost:8180     |

---

## Flujo de datos

1. El usuario introduce los datos del vuelo en la web de **Flask**, que ahora utiliza **WebSocket** para comunicarse con el servidor en tiempo real.

2. Flask envía los datos al topic de Kafka: `flight-delay-ml-request`, añadiendo un identificador único (`UUID`) a cada mensaje.

3. **Spark** lee esos datos desde Kafka, y consulta las **distancias** entre aeropuertos **desde Cassandra** (en lugar de MongoDB).

4. Spark realiza la predicción y guarda el resultado en:
   - **MongoDB**
   - **HDFS**
   - **Kafka** (`flight-delay-ml-response`)

5. La predicción vuelve a Flask mediante **WebSocket**, que escucha la respuesta asociada al UUID enviado.

6. **NiFi** también lee las predicciones desde Kafka cada 10 segundos y las guarda en un archivo `.txt`.

---

## Entrenamiento del modelo

El modelo se entrena usando un script en PySpark: `train_spark_mllib_model.py`. Este script:

- Carga un conjunto de datos históricos de vuelos.
- Preprocesa los datos con `StringIndexer`, `Bucketizer` y `VectorAssembler`.
- Entrena un modelo de clasificación `RandomForestClassifier`.
- Registra métricas y parámetros con **MLflow**.
- Guarda los modelos entrenados en la carpeta `/models`.

El entrenamiento se automatiza mediante un DAG de **Apache Airflow**:

- **Airflow** ejecuta el script como una tarea programada.
- Se integra con **MLflow** para registrar cada experimento.
- Los modelos entrenados se reutilizan por **Spark** para predecir en tiempo real.

---

## Funcionalidades implementadas (puntos del examen)

- [x] Dockerizar toda la arquitectura (Flask, Spark, Kafka, Mongo, etc.)
- [x] Publicar predicciones en Kafka y Mongo
- [x] Leer predicciones con NiFi y guardarlas cada 10s en un fichero `.txt`
- [x] Guardar predicciones también en HDFS
- [x] Entrenar el modelo desde Airflow con MLflow
- [x] Incorporar WebSocket para recibir predicciones en tiempo real
- [x] Migrar las distancias de MongoDB a Cassandra y usarlas desde Spark

---

## Autores

- [Mónica Fernández Lortal]  
- [Ana Li Camello Serrano]
