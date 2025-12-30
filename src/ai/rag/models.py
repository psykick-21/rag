from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class DocumentChunk:
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RetrievedDocumentChunk:
    chunk: DocumentChunk
    distance: float

@dataclass
class RetrievalResult:
    chunks: List[RetrievedDocumentChunk]


@dataclass
class DocumentChunkEmbedding:
    document_chunk: DocumentChunk
    embedding: List[float]
