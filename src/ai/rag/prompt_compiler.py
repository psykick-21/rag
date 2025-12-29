from typing import List
from src.ai.rag.models import DocumentChunk


class PromptCompiler:
    """
    Compiles user queries and retrieved context into
    a deterministic, policy-controlled model input.
    """

    def compile(self, query: str, context: List[DocumentChunk]) -> str:
        raise NotImplementedError
