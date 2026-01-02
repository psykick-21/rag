from src.ai.rag.models import RetrievalResult
from src.ai.rag.models import DocumentChunk, RetrievedDocumentChunk
from src.db.connection import conn

from openai import OpenAI
from pgvector.psycopg import Vector
from dotenv import load_dotenv
import json

load_dotenv()


class Retriever:
    """
    Responsible only for retrieving relevant document chunks.
    """

    def retrieve(self, query: str, top_k: int = 5, only_latest = False) -> RetrievalResult:
        """Retrieves the relevant document chunks using the OpenAI API."""

        client = OpenAI()

        query_embedding = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        with conn.cursor() as cursor:
            latest_ingestion_id = None
            if only_latest:
                cursor.execute(
                    "SELECT * FROM ingestion_metadata ORDER BY ingested_at DESC LIMIT 1"
                )
                latest_ingestion_id = cursor.fetchone()[1]

            retrieval_query = f"""
            SELECT file_name, chunk_index, content, embedding, metadata, embedding <=> %s AS distance
            FROM file_chunks
            {f"WHERE metadata->>'ingestion_id' = %s" if only_latest else ""}
            ORDER BY distance ASC
            LIMIT %s
            """
            if only_latest:
                # Convert UUID to string for JSON comparison
                query_params = (Vector(query_embedding), str(latest_ingestion_id), top_k)
            else:
                query_params = (Vector(query_embedding), top_k)
                
            cursor.execute(retrieval_query, query_params)
            fetched_chunks = cursor.fetchall()
            chunks = []
            for chunk in fetched_chunks:
                # Parse metadata JSON if it's a string, otherwise use as-is
                metadata = chunk[4]  # metadata column
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                elif metadata is None:
                    metadata = {}
                
                chunks.append(RetrievedDocumentChunk(
                    chunk=DocumentChunk(
                        content=chunk[2],  # content column
                        source=chunk[0],   # file_name column
                        metadata=metadata
                    ),
                    distance=float(chunk[5])  # distance column
                ))

        return RetrievalResult(chunks=chunks)

