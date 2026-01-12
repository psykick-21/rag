from src.ai.rag.models import RetrievalResult
from src.ai.rag.models import DocumentChunk, RetrievedDocumentChunk
from src.db.connection import conn

from openai import OpenAI
from pgvector.psycopg import Vector
from psycopg import Cursor
from dotenv import load_dotenv
import json
import uuid
from typing import List, Tuple, Any

load_dotenv()


class Retriever:
    """
    Responsible only for retrieving relevant document chunks.
    """

    def __init__(self):
        self.client = OpenAI()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        only_latest: bool = False,
        retrieve_parent_chunks: bool = False,
    ) -> RetrievalResult:
        """Retrieves the relevant document chunks using the OpenAI API."""

        query_embedding = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding


        with conn.cursor() as cursor:

            retrieval_query, query_params = self._get_retrieval_query(
                cursor=cursor,
                only_latest=only_latest,
                query_embedding=query_embedding,
                top_k=top_k,
                retrieve_parent_chunks=retrieve_parent_chunks
            )
                
            chunks = self._fetch_top_k_chunks(
                cursor=cursor,
                retrieval_query=retrieval_query,
                query_params=query_params,
            )

        relevant_or_capped_chunks = self._apply_relevance_or_capped_filter(chunks)

        return RetrievalResult(chunks=relevant_or_capped_chunks)


    def _get_retrieval_query(
        self,
        cursor: Cursor,
        only_latest: bool,
        query_embedding: List[float],
        top_k: int,
        retrieve_parent_chunks: bool,
    ) -> Tuple[str, List[Any]]:

        latest_ingestion_id = None
        if only_latest:
            cursor.execute(
                "SELECT * FROM ingestion_metadata ORDER BY ingested_at DESC LIMIT 1"
            )
            latest_ingestion_id = cursor.fetchone()[1]

        where_clause = ""
        if only_latest:
            if retrieve_parent_chunks:
                where_clause = f"WHERE ingestion_id = %s"
            else:
                where_clause = "WHERE ingestion_id = %s AND chunk_type = 'child'"
        else:
            if retrieve_parent_chunks:
                pass
            else:
                where_clause = "WHERE chunk_type = 'child'"

        retrieval_query = f"""
        SELECT
            source,
            chunk_id,
            chunk_type,
            parent_chunk_id,
            ingestion_id,
            ingested_at,
            content,
            embedding,
            metadata,
            embedding <=> %s AS distance
        FROM file_chunks
        {where_clause}
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
                # Column order from SQL query:
                # 0: source, 1: chunk_id, 2: chunk_type, 3: parent_chunk_id,
                # 4: ingestion_id, 5: ingested_at, 6: content, 7: embedding,
                # 8: metadata, 9: distance

                # Parse metadata JSONB - psycopg should parse it automatically, but handle both cases
                metadata = chunk[8]  # metadata column (JSONB)
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                elif metadata is None:
                    metadata = {}
                # If it's already a dict (from JSONB), use it as-is
                
                # Convert ingestion_id from TEXT to UUID
                ingestion_id = chunk[4]  # ingestion_id column (TEXT in DB, UUID in model)
                if isinstance(ingestion_id, str):
                    ingestion_id = uuid.UUID(ingestion_id)
                
                chunks.append(RetrievedDocumentChunk(
                    chunk=DocumentChunk(
                        content=chunk[6],  # content column
                        source=chunk[0],   # source column
                        chunk_type=chunk[2],  # chunk_type column
                        chunk_id=str(chunk[1]),  # chunk_id column
                        ingestion_id=ingestion_id,  # ingestion_id column (converted to UUID)
                        ingested_at=chunk[5],  # ingested_at column
                        parent_chunk_id=str(chunk[3]) if chunk[3] else None,  # parent_chunk_id column
                        metadata={**metadata, "chunk_index": chunk[1]}
                    ),
                    distance=float(chunk[9])  # distance column
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

