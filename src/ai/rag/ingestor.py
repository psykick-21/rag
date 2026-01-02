import uuid
from datetime import datetime
from src.ai.rag.models import DocumentChunk, DocumentChunkEmbedding
from src.db.connection import conn
from typing import List
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4


class DocumentIngestor:
    """
    Responsible for:
    - Loading raw documents
    - Chunking documents
    - Embedding chunks
    - Persisting chunks to storage

    Does NOT:
    - Handle queries
    - Perform retrieval
    - Interact with APIs at request time
    """

    def __init__(self):
        self.client = OpenAI()


    def _chunk_document_text(
        self, document_text: str,
        ingestion_id: uuid.UUID,
        ingested_at: datetime,
        chunk_size: int = 500,
        overlap: int = 50,
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
                    metadata={"chunk_index": idx, "ingestion_id": ingestion_id, "ingested_at": ingested_at}
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

    
    def _update_ingestion_metadata(self, ingestion_id: uuid.UUID, ingested_at: datetime, chunks_processed: int):
        """Updates the ingestion metadata."""

        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ingestion_metadata (ingestion_id, ingested_at, num_chunks) VALUES (%s, %s, %s)",
                (ingestion_id, ingested_at, chunks_processed)
            )
            conn.commit()


    def ingest_file(self, file_path: Path, ingestion_id = None, ingested_at = None, save_ingestion_metadata = True):
        """Ingests a file."""

        if not ingestion_id:
            ingestion_id = uuid4()
        
        if not ingested_at:
            ingested_at = datetime.now()

        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
        chunks = self._chunk_document_text(document_text, ingestion_id, ingested_at)
        embedded_chunks = self._embed_chunks(chunks)
        self._save_embeddings_to_db(embedded_chunks)

        if save_ingestion_metadata:
            self._update_ingestion_metadata(ingestion_id, ingested_at, len(chunks))

        return ingestion_id, ingested_at, len(chunks)


    def ingest_directory(self, directory_path: Path):
        """Ingests a directory of documents."""

        ingestion_id = uuid4()
        ingested_at = datetime.now()
        total_chunks = 0

        for file in tqdm(directory_path.glob("*.md"), desc=f"Processing {directory_path}", leave=False):
            _, _, chunks_processed = self.ingest_file(file, ingestion_id, ingested_at, save_ingestion_metadata=False)
            total_chunks += chunks_processed

        self._update_ingestion_metadata(ingestion_id, ingested_at, total_chunks)
        return ingestion_id, ingested_at, total_chunks


if __name__ == "__main__":
    path = Path("data/raw_docs")
    ingestor = DocumentIngestor()
    ingestor.ingest_directory(path)