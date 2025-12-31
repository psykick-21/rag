from typing import List, Tuple
from src.ai.rag.models import RetrievedDocumentChunk

class PromptCompiler:
    """
    Responsible only for constructing prompts.
    No LLM calls, no business logic.
    """

    @staticmethod
    def compile(
        context: List[RetrievedDocumentChunk],
        sub_queries: List[str],
    ) -> Tuple[str, str]:
        """
        Builds system and user prompts for grounded RAG answering.
        """

        system_prompt = (
            "You are an engineering knowledge assistant.\n"
            "You must answer questions using ONLY the provided context.\n"
            "If the answer is not explicitly present in the context, say: \"I don't know.\""
        )

        questions_list = "\n".join(
            [f"{i+1}. {sub_q}" for i, sub_q in enumerate(sub_queries)]
        )

        if not context:
            user_prompt = (
                f"Answer the following questions using ONLY the context:\n{questions_list}\n\n"
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
            f"Answer the following questions using ONLY the context:\n{questions_list}\n\n"
            "Answer each part separately. If any part cannot be answered, say \"I don't know\" for that part."
        )

        return system_prompt, user_prompt
