from src.ai.rag.retriever import Retriever
from src.ai.rag.generator import Generator

class RAGPipeline:
    """
    Orchestrates retrieval and generation.
    """

    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str) -> str:
        retrieval_result = self.retriever.retrieve(query)
        answer = self.generator.generate(
            query=query,
            context=retrieval_result.chunks
        )
        return answer
