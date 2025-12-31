import re
from typing import List

def generate_sub_queries(query: str) -> List[str]:
    """Generates sub-queries based on the main query.
    
    Splits queries that contain:
    - Multiple question marks (?)
    - Conjunctions like "and", "how", "why"
    
    Args:
        query: The input query string
        
    Returns:
        List of sub-queries
        
    Example:
        "What is X and why use it?" -> ["What is X?", "Why use it?"]
    """
    if not query or not query.strip():
        return [query] if query else []
    
    query = query.strip()
    
    # Check for multiple question marks
    question_count = query.count('?')
    if question_count > 1:
        # Split by question marks and filter out empty strings
        sub_queries = [q.strip() + '?' for q in query.split('?') if q.strip()]
        return sub_queries
    
    # Check for conjunctions that indicate multiple questions
    # Split on all "and" conjunctions (case-insensitive)
    parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
    
    if len(parts) > 1:
        sub_queries = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Capitalize first letter if it's not already capitalized
            if part and not part[0].isupper():
                part = part[0].upper() + part[1:]
            
            # Ensure it ends with ?
            if not part.endswith('?'):
                part += '?'
            
            sub_queries.append(part)
        
        if len(sub_queries) > 1:
            return sub_queries
    
    # Check for "how" or "why" in the middle (not at start)
    # This handles cases like "What is X? How does it work?"
    if re.search(r'\?.*\b(how|why)\s+', query, re.IGNORECASE):
        # Already handled by multiple ? check above
        pass
    
    # No splitting needed, return original query
    return [query]


if __name__ == "__main__":

    queries = [
        # Test 1: Original example - "and" followed by "why"
        "What is FastAPI and why use it?",

        # Test 2: Multiple question marks
        "What is RAG? How does it work? Why is it useful?",

        # Test 3: "and" followed by "how"
        "Explain vector databases and how do they store embeddings?",

        # Test 4: Multiple "and" conjunctions
        "What is LangChain and what are its features and how to install it?",

        # Test 5: Simple query (should not split)
        "What is a vector database?",
    ]

    for query in queries:
        sub_queries = generate_sub_queries(query)
        print(f"Query: {query}")
        print(f"Sub-queries: {sub_queries}")
        print("-" * 100)