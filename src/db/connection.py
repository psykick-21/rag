import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(
    dbname="rag",
    user="psykick",          # usually your mac username
    password=None,           # empty for local Homebrew installs
    host="localhost",
    port=5432,
)

register_vector(conn)