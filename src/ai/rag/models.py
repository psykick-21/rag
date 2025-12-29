from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class DocumentChunk:
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    chunks: List[DocumentChunk]

@dataclass
class DocumentChunkEmbedding:
    document_chunk: DocumentChunk
    embedding: List[float]
