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

    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4.1-nano"

    def generate_response(self, context: List[RetrievedDocumentChunk], sub_queries: List[str]) -> ChatCompletion:
        """Generates an answer strictly using the provided context."""

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
