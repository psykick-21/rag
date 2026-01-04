from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


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



class AnswerEvaluation(BaseModel):
    """
    Structured self-evaluation of a generated RAG answer.

    This model is produced by an evaluation LLM call and is used
    for debugging, observability, and future decision-making.
    """

    grounded: bool = Field(
        ...,
        description=(
            "True if the answer is fully supported by the provided context. "
            "False if the answer includes information not present in the retrieved chunks."
        ),
    )

    sufficient_context: bool = Field(
        ...,
        description=(
            "True if the retrieved context was sufficient to answer the question completely. "
            "False if important information appears to be missing from the context."
        ),
    )

    missing_aspects: List[str] = Field(
        default_factory=list,
        description=(
            "Specific aspects of the question that were not answered due to missing "
            "or insufficient context. Empty if the answer is complete."
        ),
    )

    confidence_alignment: bool = Field(
        ...,
        description=(
            "True if the system's confidence label (low/medium/high) matches the actual "
            "quality and completeness of the answer. False if the confidence appears "
            "overstated or understated."
        ),
    )