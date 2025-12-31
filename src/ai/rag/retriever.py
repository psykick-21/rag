from src.ai.rag.models import RetrievalResult
from src.ai.rag.models import DocumentChunk, DocumentChunkEmbedding, RetrievedDocumentChunk
from src.db.connection import conn

from typing import List
from openai import OpenAI
from pgvector.psycopg import Vector
from dotenv import load_dotenv
import os
from pathlib import Path
from tqdm import tqdm

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


    def _chunk_document_text(
        self, document_text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[DocumentChunk]:
        """Chunks the document text into smaller chunks of the given size."""

        chunks = []
        idx = 0

        for i in range(0, len(document_text), chunk_size - overlap):
            chunk_text = document_text[i:i + chunk_size]
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    source="document",
                    metadata={"chunk_index": idx}
                )
            )
            idx += 1
        return chunks

    
    def _embed_chunks(
        self,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunkEmbedding]:
        """Embeds the chunks using the OpenAI API."""

        client = OpenAI()

        embeded_chunks = []

        for chunk in tqdm(chunks, desc="Embedding chunks", leave=False):
            embedding = client.embeddings.create(
                input=chunk.content,
                model="text-embedding-3-small"
            )
            embeded_chunk = DocumentChunkEmbedding(
                document_chunk=chunk,
                embedding=embedding.data[0].embedding
            )
            embeded_chunks.append(embeded_chunk)
        
        return embeded_chunks


    def _save_embeddings_to_db(
        self,
        embeddings: List[DocumentChunkEmbedding]
    ):
        """Saves the embeddings to the database."""

        with conn.cursor() as cursor:
            for embedding in tqdm(embeddings, desc="Saving embeddings to database", leave=False):
                cursor.execute(
                    "INSERT INTO file_chunks (file_name, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                    (embedding.document_chunk.source, embedding.document_chunk.metadata["chunk_index"], embedding.document_chunk.content, embedding.embedding)
                )
            conn.commit()


if __name__ == "__main__":
    retriever = Retriever()
    raw_docs_path = Path("data/raw_docs")
    
    # Collect all directories first
    directories = [d for d in raw_docs_path.iterdir() if d.is_dir()]
    
    # Walk through all directories in /data/raw_docs
    for directory in tqdm(directories, desc="Processing directories"):
        directory_name = directory.name
        
        # Find all markdown files in the directory
        md_files = list(directory.glob("*.md"))
        
        for md_file in tqdm(md_files, desc=f"Processing {directory_name}", leave=False):
            file_name = md_file.name
            
            # Read the markdown file
            with open(md_file, "r", encoding="utf-8") as f:
                document_text = f.read()
            
            # Create source name as directory_name + file_name
            source_name = f"{directory_name}/{file_name}"
            
            # Chunk the document
            chunks = retriever._chunk_document_text(document_text)
            
            # Update source for each chunk
            for chunk in chunks:
                chunk.source = source_name
            
            # Embed the chunks
            embedded_chunks = retriever._embed_chunks(chunks)
            
            # Save to database
            retriever._save_embeddings_to_db(embedded_chunks)

