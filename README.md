# Engineering Knowledge RAG Assistant

## Problem
Engineering teams accumulate large amounts of technical documentation (APIs, design docs, READMEs), but finding precise answers quickly is hard. This project builds a RAG-based chatbot that answers developer questions using internal engineering documents as its knowledge source.

## User Flow
1. Documents are ingested and indexed
2. User asks a technical question in natural language
3. Relevant document chunks are retrieved
4. LLM generates an answer grounded in retrieved context
5. Sources are shown alongside the answer

## Non-Goals
- Not building a general-purpose chatbot
- Not optimizing for research-level accuracy
- Not supporting multiple document types initially
- Not focusing on UI polish

## Setup

1. Install dependencies using `uv`:
```bash
uv sync
```

2. Set up your environment variables (database connection, OpenAI API key, etc.)

3. Ingest your documents:
```bash
# Add your documents to data/raw_docs/
# Then run the ingestion process
```

## Running

Start the API server:
```bash
python -m src.api.app
```

Or run directly with uvicorn:
```bash
uvicorn src.api.app:app --reload
```

The API will be available at `http://localhost:8000`

**Note:** The API includes CORS middleware and request/response logging middleware for development and debugging.

## API Endpoints

### Chat Endpoint
`GET /api/v1/chat` - Ask a question and get an answer with citations

**Query Parameters:**
- `query` (required) - The user query string
- `only_latest` (optional, default: `false`) - Whether to return only the latest results
- `debug` (optional, default: `false`) - Whether to return debug information including sub-queries, retrieval metrics, token usage, and evaluation results
- `use_llm_decomposition` (optional, default: `false`) - Whether to use LLM-based query decomposition instead of manual rule-based decomposition

**Response Format:**
```json
{
  "answer": "The generated answer...",
  "citations": [
    {
      "source": "path/to/document.md",
      "chunk_index": 0
    }
  ],
  "confidence": "medium",
  "debug": { ... },  // Only included if debug=true
  "evaluation": "..."  // Only included if debug=true
}
```

### Other Endpoints
- `GET /api/v1/ingestions` - View ingestion history (returns ingestion_id, timestamp, and number_of_chunks)
- `GET /health` - Health check endpoint

## Project Structure

```
rag/
├── pyproject.toml          # Project dependencies and configuration
├── uv.lock                 # Lock file for dependency versions
├── README.md               # Project documentation
│
├── data/
│   └── raw_docs/           # Source documentation files
│       ├── airflow/        # Airflow documentation
│       ├── awesome_genai/  # Awesome GenAI documentation
│       ├── baml/           # BAML documentation
│       ├── fastapi/        # FastAPI documentation
│       ├── langchain/      # LangChain documentation
│       ├── pgvector/       # pgvector documentation
│       └── tanstack/       # TanStack documentation
│
├── src/
│   ├── ai/
│   │   ├── __init__.py
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── evaluator.py        # Answer evaluation and quality assessment
│   │       ├── generator.py        # LLM response generation
│   │       ├── ingestor.py         # Document ingestion and indexing
│   │       ├── models.py           # Data models and schemas
│   │       ├── orchestrator.py     # RAG orchestration logic
│   │       ├── prompt_compiler.py  # Prompt construction
│   │       ├── query_analyzer.py   # Query analysis and processing
│   │       ├── retriever.py        # Document retrieval logic
│   │       └── utils/
│   │           ├── __init__.py
│   │           ├── confidence.py       # Confidence scoring utilities
│   │           ├── debug_utils.py      # Debugging utilities
│   │           └── retriever_utils.py  # Retriever helper functions
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI application setup
│   │   │
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── health.py           # Health check endpoint
│   │   │   ├── chat.py             # Chat API endpoint
│   │   │   └── ingestions.py       # Ingestion history endpoint
│   │   │
│   │   ├── models/
│   │   │   └── __init__.py         # Pydantic schemas
│   │   │
│   │   ├── utils/
│   │   │   └── __init__.py
│   │   │
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── cors.py             # CORS configuration
│   │       └── logging.py          # Request/response logging
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   └── connection.py           # Database connection setup
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger/
│           ├── __init__.py
│           └── logger.py          # Logging utilities
│
└── docs/
    ├── daily_logs.md               # Development logs
    ├── Explanations.md             # Technical documentation
    ├── failure_modes.md            # Failure mode documentation
    └── ingestion_assumptions.md    # Document ingestion assumptions
```

## RAG Components

### Core Components (`src/ai/rag/`)

- **`orchestrator.py`** - Coordinates the RAG pipeline: query analysis, retrieval, context assembly, generation, and confidence computation. Main entry point for processing queries.

- **`ingestor.py`** - Handles document ingestion: loads raw documents, chunks them into smaller pieces, generates embeddings using OpenAI, and persists chunks to the database. Does not handle queries or retrieval.

- **`retriever.py`** - Retrieves relevant document chunks using vector similarity search. Embeds the query and searches the database for the most similar chunks based on cosine distance.

- **`generator.py`** - Generates answers grounded in retrieved document context. Uses OpenAI's chat completion API with strict rules to only use provided context and avoid hallucination.

- **`evaluator.py`** - Evaluates answer quality and relevance using structured output. Provides automated assessment of answer accuracy, completeness, and grounding in retrieved context. Returns evaluation results including grounded status, sufficient context check, missing aspects, and confidence alignment.

- **`query_analyzer.py`** - Analyzes user queries and splits complex questions into sub-queries. Supports two modes:
  - **Manual decomposition**: Rule-based splitting for multiple question marks, conjunctions like "and", and questions containing "how" or "why"
  - **LLM-based decomposition**: Uses GPT-4.1-nano to intelligently decompose complex queries into sub-queries

- **`prompt_compiler.py`** - Constructs system and user prompts for the LLM. Formats retrieved context chunks and sub-queries into structured prompts for grounded answering.

- **`models.py`** - Defines data models: `DocumentChunk`, `RetrievedDocumentChunk`, `RetrievalResult`, and `DocumentChunkEmbedding`.

### Utility Components (`src/ai/rag/utils/`)

- **`retriever_utils.py`** - Helper functions for retrieval: deduplicates retrieved chunks and filters top-k chunks based on distance scores.

- **`confidence.py`** - Computes confidence levels (low/medium/high) based on retrieval distance scores to assess answer reliability.

- **`debug_utils.py`** - Debugging utilities for development and troubleshooting.
