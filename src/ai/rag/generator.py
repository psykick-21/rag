from typing import List
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from src.ai.rag.models import RetrievedDocumentChunk
from src.ai.rag.prompt_compiler import PromptCompiler
from src.utils.logger import getLogger

logger = getLogger(__name__)

class Generator:
    """
    Responsible only for generating an answer grounded
    in retrieved document context.
    """

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.client = OpenAI()
        self.model = model

    def generate_response(self, context: List[RetrievedDocumentChunk], sub_queries: List[str]) -> ChatCompletion:
        """
        Generate an answer strictly using the provided context.

        Rules:
        - Use ONLY the given context chunks
        - If the answer is not present, say "I don't know"
        - Do not hallucinate or add external knowledge
        """

        system_prompt, user_prompt = PromptCompiler.compile(context, sub_queries)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        
        return response
