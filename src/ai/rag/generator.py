from typing import List
from openai import OpenAI
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

    def generate_answer(self, query: str, context: List[RetrievedDocumentChunk]) -> str:
        """
        Generate an answer strictly using the provided context.

        Rules:
        - Use ONLY the given context chunks
        - If the answer is not present, say "I don't know"
        - Do not hallucinate or add external knowledge
        """

        system_prompt, user_prompt = PromptCompiler.compile(query, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        model_used = response.model

        logger.info(f"Generated answer for query: \"{query}\" using model: {model_used} with {input_tokens} input tokens and {output_tokens} output tokens")

        answer = response.choices[0].message.content.strip()
        
        return answer
