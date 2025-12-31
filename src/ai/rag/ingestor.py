from src.ai.rag.models import DocumentChunk, DocumentChunkEmbedding
from src.db.connection import conn
from typing import List
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path


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


    def ingest_file(self, file_path: Path):
        """Ingests a file."""

        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
        chunks = self._chunk_document_text(document_text)
        embedded_chunks = self._embed_chunks(chunks)
        self._save_embeddings_to_db(embedded_chunks)


    def ingest_directory(self, directory_path: Path):
        """Ingests a directory of documents."""

        for file in tqdm(directory_path.glob("*.md"), desc=f"Processing {directory_path}", leave=False):
            self.ingest_file(file)
            


if __name__ == "__main__":
    path = Path("data/raw_docs")
    ingestor = DocumentIngestor()
    ingestor.ingest_directory(path)