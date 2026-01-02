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

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /api/v1/chat?query=your_question&only_latest=false` - Ask a question
- `GET /api/v1/ingestions` - View ingestion history
- `GET /health` - Health check

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

- **`query_analyzer.py`** - Analyzes user queries and splits complex questions into sub-queries. Handles multiple question marks, conjunctions like "and", and questions containing "how" or "why".

- **`prompt_compiler.py`** - Constructs system and user prompts for the LLM. Formats retrieved context chunks and sub-queries into structured prompts for grounded answering.

- **`models.py`** - Defines data models: `DocumentChunk`, `RetrievedDocumentChunk`, `RetrievalResult`, and `DocumentChunkEmbedding`.

### Utility Components (`src/ai/rag/utils/`)

- **`retriever_utils.py`** - Helper functions for retrieval: deduplicates retrieved chunks and filters top-k chunks based on distance scores.

- **`confidence.py`** - Computes confidence levels (low/medium/high) based on retrieval distance scores to assess answer reliability.
