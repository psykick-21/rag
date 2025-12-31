from src.ai.rag.retriever import Retriever
from src.ai.rag.generator import Generator
from src.ai.rag.query_analyzer import generate_sub_queries
from src.ai.rag.utils.retriever_utils import dedupe_retrieved_chunks, filter_top_k_chunks
from src.ai.rag.utils.confidence import compute_confidence
from src.utils.logger import getLogger

logger = getLogger(__name__)

class RAGOrchestrator:
    """
    Responsible for coordinating the RAG pipeline:
    - Query analysis
    - Retrieval
    - Context assembly
    - Generation
    - Confidence computation

    Does NOT:
    - Call APIs directly from routes
    - Contain prompt logic
    - Contain retrieval logic
    - Ingest data
    """

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def run(
        self,
        query: str,
    ) -> str:
        
        sub_queries = generate_sub_queries(query)
        logger.info(f"Generated {len(sub_queries)} sub-queries for query: {query}")
        
        retrieval_results = []
        for sub_query in sub_queries:
            retrieval_result = self.retriever.retrieve(sub_query)
            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for sub-query: {sub_query}")
            retrieval_results.extend(retrieval_result.chunks)

        deduplicated_retrieval_chunks = dedupe_retrieved_chunks(retrieval_results)
        logger.info(f"Deduplicated chunks from {len(retrieval_results)} to {len(deduplicated_retrieval_chunks)}")

        filtered_chunks = filter_top_k_chunks(deduplicated_retrieval_chunks)
        logger.info(f"Filtered chunks from {len(deduplicated_retrieval_chunks)} to {len(filtered_chunks)}")
        
        response = self.generator.generate_response(filtered_chunks, sub_queries)
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        model_used = response.model
        query_str = " | ".join(sub_queries)
        logger.info(f"Generated answer for query: \"{query_str}\" using model: {model_used} with {input_tokens} input tokens and {output_tokens} output tokens")

        answer = response.choices[0].message.content.strip()
        
        citations = [
            {
                "source": chunk.chunk.source,
                "chunk_index": chunk.chunk.metadata["chunk_index"]
            }
            for chunk in filtered_chunks
        ]

        scores = [chunk.distance for chunk in filtered_chunks]
        confidence = compute_confidence(scores)
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
