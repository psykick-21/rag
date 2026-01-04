from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from src.ai.rag.models import AnswerEvaluation, RetrievedDocumentChunk, DocumentChunk
from src.ai.rag.prompt_compiler import PromptCompiler

load_dotenv()

class ResponseEvaluator:
    """
    Responsible for evaluating the response.
    """

    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-4.1-nano"

    def evaluate(
        self,
        query: str,
        context: List[RetrievedDocumentChunk],
        answer: str
    ) -> AnswerEvaluation:
        """Evaluates the response."""

        system_prompt, user_prompt = PromptCompiler.compile_evaluation_prompt(
            query=query,
            context=context,
            answer=answer
        )
        
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=AnswerEvaluation,
            temperature=0.0
        )
        return response.choices[0].message.parsed