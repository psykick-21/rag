from src.ai.rag.models import RetrievalResult
from src.ai.rag.models import DocumentChunk, RetrievedDocumentChunk
from src.db.connection import conn

from openai import OpenAI
from pgvector.psycopg import Vector
from dotenv import load_dotenv

load_dotenv()


class Retriever:
    """
    Responsible only for retrieving relevant document chunks.
    """

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieves the relevant document chunks using the OpenAI API."""

        client = OpenAI()

        query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *, embedding <=> %s AS distance
                FROM file_chunks
                ORDER BY distance ASC
                LIMIT %s
                """,
                (Vector(query_embedding), top_k)
            )
            fetched_chunks = cursor.fetchall()
            chunks = [RetrievedDocumentChunk(
                chunk=DocumentChunk(
                content=chunk[3],
                    source=chunk[1],
                    metadata={"chunk_index": chunk[2]}
                ),
                distance=chunk[6]
            ) for chunk in fetched_chunks]

        return RetrievalResult(chunks=chunks)

