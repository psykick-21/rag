from src.ai.rag.models import (
    DocumentChunk,
    DocumentChunkEmbedding,
    ParentDocumentChunk,
    ChildDocumentChunk,
)
from src.db.connection import conn
from src.ai.rag.utils.ingestor_utils import make_json_serializable

import uuid
from datetime import datetime
from typing import List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()


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


    def ingest_file(
        self,
        file_path: Path,
        ingestion_id: Optional[uuid.UUID] = None,
        ingested_at: Optional[datetime] = None,
        save_ingestion_metadata: bool = True,
    ) -> Tuple[uuid.UUID, datetime, int]:
        """
        Ingests a single file by reading its contents, chunking the document,
        embedding all resulting chunks, and saving the embeddings to the database.

        Optionally updates ingestion metadata if requested.

        Args:
            file_path (Path): The path to the file to ingest.
            ingestion_id (Optional[uuid.UUID]): The unique ID for this ingestion. Generated externally if not provided.
            ingested_at (Optional[datetime]): The timestamp for this ingestion. Set externally if not provided.
            save_ingestion_metadata (bool): Whether to update/save ingestion metadata after processing this file.

        Returns:
            Tuple[uuid.UUID, datetime, int]: The ingestion_id, ingested_at timestamp, and the number of chunks processed.
        """

        with open(file_path, "r", encoding="utf-8") as f:
            document_text = f.read()
        document_chunks = self.chunk_document_text(document_text, str(file_path), ingestion_id, ingested_at)
        embedded_chunks = self.embed_chunks(document_chunks)
        self.save_embeddings_to_db(embedded_chunks)
    
        if save_ingestion_metadata:
            self.update_ingestion_metadata(ingestion_id, ingested_at, len(document_chunks))

        return ingestion_id, ingested_at, len(document_chunks)


    def ingest_directory(
        self,
        directory_path: Path
    ) -> Tuple[uuid.UUID, datetime, int]:
        """
        Ingests all Markdown (.md) files in a specified directory.

        This method iterates through all .md files in the given directory,
        ingests each file (chunking, embedding, and saving embeddings to the database),
        and aggregates the total number of processed chunks. A single ingestion ID 
        and timestamp is used for the entire batch, and ingestion metadata is updated
        once after all files are processed.

        Args:
            directory_path (Path): The path to the directory containing documents to ingest.

        Returns:
            Tuple[uuid.UUID, datetime, int]: The ingestion_id, ingested_at timestamp, 
                and the total number of chunks processed across all files.
        """
        ingestion_id = uuid4()
        ingested_at = datetime.now()
        total_chunks = 0

        for file in tqdm(directory_path.glob("*.md"), desc=f"Processing {directory_path}", leave=False):
            _, _, chunks_processed = self.ingest_file(file, ingestion_id, ingested_at, save_ingestion_metadata=False)
            total_chunks += chunks_processed

        self.update_ingestion_metadata(ingestion_id, ingested_at, total_chunks)
        return ingestion_id, ingested_at, total_chunks


    def chunk_document_text(
        self,
        document_text: str,
        source: str,
        ingestion_id: uuid.UUID,
        ingested_at: datetime,
    ) -> List[DocumentChunk]:
        """Chunks the document text into parent sections."""

        parent_chunks = self._divide_document_text_into_parent_sections(document_text, source)

        all_child_chunks = []
        for parent_chunk in parent_chunks:
            child_chunks = self._chunk_parent_sections(
                document_text=parent_chunk.content,
                source=source,
                parent_chunk_id=parent_chunk.chunk_id,
                chunk_size=500,
                overlap=50,
            )
            all_child_chunks.extend(child_chunks)

        child_document_chunks = self._convert_child_chunks_to_document_chunks(all_child_chunks, ingestion_id, ingested_at)
        parent_document_chunks = self._convert_parent_chunks_to_document_chunks(parent_chunks, ingestion_id, ingested_at)

        return child_document_chunks + parent_document_chunks


    def embed_chunks(
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


    def _convert_child_chunks_to_document_chunks(
        self,
        child_chunks: List[ChildDocumentChunk],
        ingestion_id: uuid.UUID,
        ingested_at: datetime,
    ) -> List[DocumentChunk]:
        """Converts child chunks to document chunks."""

        document_chunks = []

        for child_chunk in child_chunks:
            document_chunks.append(DocumentChunk(
                content=child_chunk.content,
                source=child_chunk.source,
                chunk_type="child",
                chunk_id=f"p{child_chunk.parent_chunk_id}-c{child_chunk.chunk_id}",
                parent_chunk_id=f"p{child_chunk.parent_chunk_id}",
                ingestion_id=ingestion_id,
                ingested_at=ingested_at,
            ))
        return document_chunks
    

    def _convert_parent_chunks_to_document_chunks(
        self,
        parent_chunks: List[ParentDocumentChunk],
        ingestion_id: uuid.UUID,
        ingested_at: datetime,
    ) -> List[DocumentChunk]:
        """Converts parent chunks to document chunks."""

        document_chunks = []
        for parent_chunk in parent_chunks:
            document_chunks.append(DocumentChunk(
                content=parent_chunk.content,
                source=parent_chunk.source,
                chunk_type="parent",
                chunk_id=f"p{parent_chunk.chunk_id}",
                parent_chunk_id=None,
                ingestion_id=ingestion_id,
                ingested_at=ingested_at,
            ))
        return document_chunks


    def _divide_document_text_into_parent_sections(
        self,
        document_text: str,
        source: str,
    ) -> List[ParentDocumentChunk]:
        """
        Divides markdown text into parent sections based on boundaries:
        - Boundaries are: level 1 headings (#), level 2 headings (##), and line separators (---, ***, ___)
        - Everything between boundaries becomes a parent section
        - Content before the first boundary also becomes a parent section
        - Headings are included in their sections; line separators are skipped
        """

        lines = document_text.splitlines()
        section_chunks = []
        current_section = []

        def finish_section():
            """Finish and save the current section."""
            nonlocal current_section
            if current_section:
                section_chunks.append('\n'.join(current_section).strip())
                current_section = []

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            
            # Check if this line is a boundary
            if self._is_boundary(line):
                # Finish the current section (if any)
                finish_section()
                
                # If it's a heading, start a new section with the heading
                stripped = line.strip()
                if stripped.startswith("# ") or stripped.startswith("## "):
                    current_section = [line]
                    idx += 1
                    
                    # Collect content until we hit another boundary
                    # Include all sub-headings (###, ####, etc.) in the current section
                    while idx < len(lines):
                        next_line = lines[idx]
                        # Stop at next boundary (h1, h2, or line separator)
                        if self._is_boundary(next_line):
                            break
                        current_section.append(next_line)
                        idx += 1
                    
                    finish_section()
                    continue
                else:
                    # It's a line separator - skip it and start a new section
                    idx += 1
                    continue
            
            # Not a boundary - add to current section
            current_section.append(line)
            idx += 1
        
        # Finish any remaining section
        finish_section()

        return [ParentDocumentChunk(
            content=chunk,
            source=source,
            chunk_id=i,
        ) for i, chunk in enumerate(section_chunks) if chunk]


    def _chunk_parent_sections(
        self, 
        document_text: str,
        source: str,
        parent_chunk_id: int,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[ChildDocumentChunk]:
        """Chunks the document text into smaller chunks of the given size."""

        child_chunks = []
        idx = 0

        for i in range(0, len(document_text), chunk_size - overlap):
            chunk_text = document_text[i:i + chunk_size]
            child_chunks.append(
                ChildDocumentChunk(
                    content=chunk_text,
                    source=source,
                    parent_chunk_id=parent_chunk_id,
                    chunk_id=idx,
                )
            )
            idx += 1
        return child_chunks


    def _is_line_separator(self, line: str) -> bool:
        """Check if a line is a horizontal rule separator (---, ***, ___)."""
        stripped = line.strip()
        if not stripped:
            return False
        # Check if line consists of at least 3 of the same character: -, *, or _
        if len(stripped) >= 3:
            first_char = stripped[0]
            if first_char in ['-', '*', '_']:
                return all(c == first_char for c in stripped)
        return False


    def _is_boundary(self, line: str) -> bool:
        """Check if a line is a boundary (h1, h2, or line separator)."""
        stripped = line.strip()
        return (
            stripped.startswith("# ") or 
            stripped.startswith("## ") or 
            self._is_line_separator(line)
        )

    
    def save_embeddings_to_db(
        self,
        embeddings: List[DocumentChunkEmbedding]
    ):
        """Saves the embeddings to the database using the new table schema."""

        with conn.cursor() as cursor:
            for embedding in tqdm(embeddings, desc="Saving embeddings to database", leave=False):
                doc_chunk = embedding.document_chunk
                serializable_metadata = make_json_serializable(doc_chunk.metadata)
                cursor.execute(
                    """
                    INSERT INTO file_chunks (
                        content,
                        source,
                        chunk_id,
                        chunk_type,
                        parent_chunk_id,
                        intestion_id,
                        ingested_at,
                        embedding,
                        metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        doc_chunk.content,
                        doc_chunk.source,
                        doc_chunk.chunk_id,
                        doc_chunk.chunk_type,
                        doc_chunk.parent_chunk_id,
                        doc_chunk.ingestion_id,
                        doc_chunk.ingested_at,
                        embedding.embedding,
                        serializable_metadata
                    )
                )
            conn.commit()

    
    def update_ingestion_metadata(
        self,
        ingestion_id: uuid.UUID,
        ingested_at: datetime,
        chunks_processed: int,
    ):
        """Updates the ingestion metadata."""

        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ingestion_metadata (ingestion_id, ingested_at, chunks_processed) VALUES (%s, %s, %s)",
                (ingestion_id, ingested_at, chunks_processed)
            )
            conn.commit()


if __name__ == "__main__":

    ingestor = DocumentIngestor()
    raw_docs_root = Path("data/raw_docs")

    for folder in tqdm(raw_docs_root.iterdir(), desc="Ingesting folders", leave=False):
        if folder.is_dir():
            ingestor.ingest_directory(folder)