from src.ai.rag.models import RetrievalResult
from src.ai.rag.models import DocumentChunk, RetrievedDocumentChunk
from src.db.connection import conn

from openai import OpenAI
from pgvector.psycopg import Vector
from psycopg import Cursor
from dotenv import load_dotenv
import json
from typing import List, Tuple, Any

load_dotenv()


class Retriever:
    """
    Responsible only for retrieving relevant document chunks.
    """

    def __init__(self):
        self.client = OpenAI()

    def retrieve(self, query: str, top_k: int = 10, only_latest = False) -> RetrievalResult:
        """Retrieves the relevant document chunks using the OpenAI API."""

        query_embedding = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding


        with conn.cursor() as cursor:

            retrieval_query, query_params = self._get_retrieval_query(
                cursor,
                only_latest,
                query_embedding,
                top_k
            )
                
            chunks = self._fetch_top_k_chunks(cursor, retrieval_query, query_params)

        relevant_or_capped_chunks = self._apply_relevance_or_capped_filter(chunks)

        return RetrievalResult(chunks=relevant_or_capped_chunks)


    def _get_retrieval_query(
        self,
        cursor: Cursor,
        only_latest: bool,
        query_embedding: List[float],
        top_k: int
    ) -> Tuple[str, List[Any]]:

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

        return retrieval_query, query_params


    def _fetch_top_k_chunks(
        self,
        cursor: Cursor,
        retrieval_query: str,
        query_params: Tuple[Any]
    ) -> List[RetrievedDocumentChunk]:

        cursor.execute(retrieval_query, query_params)
        fetched_chunks = cursor.fetchall()

        chunks = []
        if fetched_chunks:
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
                        metadata={**metadata, "chunk_index": chunk[1]}
                    ),
                    distance=float(chunk[5])  # distance column
                ))
        
        return chunks


    def _apply_relevance_or_capped_filter(
        self,
        chunks: List[RetrievedDocumentChunk],
        relevance_threshold_distance: float = 0.5
    ) -> List[RetrievedDocumentChunk]:
        """Applies the relevance filter to the chunks."""

        relevant_chunks = [chunk for chunk in chunks if chunk.distance < relevance_threshold_distance]
        capped_chunks = sorted(relevant_chunks, key=lambda x: x.distance, reverse=False)[:5]
        return capped_chunks

