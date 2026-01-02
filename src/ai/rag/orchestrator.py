from src.ai.rag.retriever import Retriever
from src.ai.rag.generator import Generator
from src.ai.rag.query_analyzer import generate_sub_queries
from src.ai.rag.utils.retriever_utils import dedupe_retrieved_chunks, filter_top_k_chunks
from src.ai.rag.utils.confidence import compute_confidence
from src.ai.rag.utils.debug_utils import DebugUtils
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
        only_latest: bool = False,
        debug: bool = False
    ) -> str:

        debug_payload = {
            "query": query,
            "only_latest": only_latest,
            "sub_queries": []
        }
        
        # STEP 1: DECOMPOSE THE QUERY INTO SUB-QUERIES
        sub_queries = generate_sub_queries(query)
        logger.info(f"Sub-queries generated = {len(sub_queries)}")
        
        # STEP 2: RETRIEVE THE CHUNKS FOR EACH SUB-QUERY
        retrieval_results = []
        for sub_query in sub_queries:
            retrieval_result = self.retriever.retrieve(sub_query, only_latest=only_latest)
            logger.info(f"Retrieved {len(retrieval_result.chunks)} chunks for sub-query: {sub_query}")
            retrieval_results.extend(retrieval_result.chunks)
            debug_payload["sub_queries"].append(DebugUtils.calc_debug_metrics_for_sub_query(sub_query, retrieval_result))
        logger.info(f"Retrieved chunks = {len(retrieval_results)}")

        debug_payload["retrieved_chunks"] = len(retrieval_results)

        # STEP 3: DEDUPLICATE THE CHUNKS
        deduplicated_retrieval_chunks = dedupe_retrieved_chunks(retrieval_results)
        logger.info(f"Deduplicated chunks = {len(deduplicated_retrieval_chunks)}")
        
        debug_payload["deduplicated_chunks"] = len(deduplicated_retrieval_chunks)
        
        # STEP 4: GENERATE THE ANSWER
        response = self.generator.generate_response(deduplicated_retrieval_chunks, sub_queries)
        answer = response.choices[0].message.content.strip()
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        model_used = response.model
        logger.info(f"Generated answer for the query.\nModel = {model_used}. Input tokens = {input_tokens}. Output tokens = {output_tokens}")

        debug_payload["input_tokens"] = input_tokens
        debug_payload["output_tokens"] = output_tokens
        debug_payload["model"] = model_used

        # STEP 5: COMPUTE THE CITATIONS
        citations = [
            {
                "source": chunk.chunk.source,
                "chunk_index": chunk.chunk.metadata["chunk_index"]
            }
            for chunk in deduplicated_retrieval_chunks
        ]

        # STEP 6: COMPUTE THE CONFIDENCE
        scores = [chunk.distance for chunk in deduplicated_retrieval_chunks]
        confidence = compute_confidence(scores)
        
        # STEP 7: RETURN THE ANSWER, CITATIONS, AND CONFIDENCE
        if debug:
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "debug": debug_payload
            }
        else:
            return {
                "answer": answer,
                "citations": citations,
                "confidence": confidence
            }
