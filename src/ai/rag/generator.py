from typing import List
from openai import OpenAI
from src.ai.rag.models import DocumentChunk
from src.ai.rag.prompt_compiler import PromptCompiler


class Generator:
    """
    Responsible only for generating an answer grounded
    in retrieved document context.
    """

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.client = OpenAI()
        self.model = model

    def generate_answer(self, query: str, context: List[DocumentChunk]) -> str:
        """
        Generate an answer strictly using the provided context.

        Rules:
        - Use ONLY the given context
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

        answer = response.choices[0].message.content.strip()
        
        return answer
