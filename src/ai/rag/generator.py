from src.ai.rag.models import DocumentChunk
from typing import List

class Generator:
    """
    Responsible for generating an answer grounded in retrieved context.
    """

    def generate(
        self,
        query: str,
        context: List[DocumentChunk]
    ) -> str:
        raise NotImplementedError
