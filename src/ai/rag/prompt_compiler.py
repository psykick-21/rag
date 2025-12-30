from typing import List, Tuple
from src.ai.rag.models import RetrievedDocumentChunk

class PromptCompiler:
    """
    Responsible only for constructing prompts.
    No LLM calls, no business logic.
    """

    @staticmethod
    def compile(
        query: str,
        context: List[RetrievedDocumentChunk],
    ) -> Tuple[str, str]:
        """
        Builds system and user prompts for grounded RAG answering.
        """

        system_prompt = (
            "You are an engineering knowledge assistant.\n"
            "You must answer questions using ONLY the provided context.\n"
            "If the answer is not explicitly present in the context, say: \"I don't know.\""
        )

        if not context:
            user_prompt = (
                f"Question:\n{query}\n\n"
                "There is no available context.\n"
                "Answer: I don't know."
            )
            return system_prompt, user_prompt

        context_block = "\n\n".join(
            [
                f"Document Chunk {i+1}:\n{chunk.chunk.content}"
                for i, chunk in enumerate(context)
            ]
        )

        user_prompt = (
            f"{context_block}\n\n"
            f"Question:\n{query}\n\n"
            "Answer the question using only the context above."
        )

        return system_prompt, user_prompt
