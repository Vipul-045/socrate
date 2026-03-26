import psycopg2
from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def connect_db():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def create_table():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR(1536)
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_document(content, embedding):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, embedding)
    )
    conn.commit()
    cur.close()
    conn.close()

def search_similar(embedding):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT content
        FROM documents
        ORDER BY embedding <-> %s
        LIMIT 3;
    """, (embedding,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in results]