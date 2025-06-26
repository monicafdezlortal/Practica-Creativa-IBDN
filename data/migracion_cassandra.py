from cassandra.cluster import Cluster, OperationTimedOut
import time, json

for i in range(20):
    try:
        cluster = Cluster(['cassandra'], port=9042)
        session = cluster.connect()
        break
    except Exception as e:
        print(f"Intento {i+1}: Cassandra aún no responde. Esperando...")
        time.sleep(10)
else:
    print("No se pudo conectar a Cassandra después de varios intentos.")
    exit(1)

print("Cassandra lista, ejecutando migración...")

# --- Espera adicional para estabilidad interna del keyspace ---
time.sleep(25)

# --- Crear keyspace y tabla ---
try:
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS flights
        WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
    """)
    session.set_keyspace('flights')
    session.execute("""
        CREATE TABLE IF NOT EXISTS origin_dest_distances (
            origin text,
            dest text,
            distance double,
            PRIMARY KEY (origin, dest)
        )
    """)
except OperationTimedOut:
    print("Timeout al crear keyspace o tabla. Intenta aumentar el tiempo de espera.")
    exit(1)

# --- Cargar datos desde JSONL ---
try:
    with open("/app/origin_dest_distances.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            origin = doc.get("Origin", "")
            dest = doc.get("Dest", "")
            distance = float(doc.get("Distance", 0.0))

            session.execute(
                "INSERT INTO origin_dest_distances (origin, dest, distance) VALUES (%s, %s, %s)",
                (origin, dest, distance)
            )
    print("Datos migrados correctamente a Cassandra.")
except Exception as e:
    print(f"Error cargando datos: {e}")
