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

## Project Structure

```
rag/
├── main.py                 # Application entry point
├── pyproject.toml          # Project dependencies and configuration
├── README.md               # Project documentation
│
├── src/
│   ├── ai/
│   │   └── rag/
│   │       ├── generator.py        # LLM response generation
│   │       ├── models.py            # Data models and schemas
│   │       ├── orchestrator.py     # RAG orchestration logic
│   │       ├── prompt_compiler.py  # Prompt construction
│   │       └── retriever.py        # Document retrieval logic
│   │
│   └── api/
│       ├── app.py                  # FastAPI application setup
│       ├── __init__.py
│       │
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── health.py           # Health check endpoint
│       │   └── chat.py             # Chat API endpoint
│       │
│       ├── models/
│       │   └── __init__.py         # Pydantic schemas
│       │
│       ├── utils/
│       │   └── __init__.py         # Utility functions
│       │
│       └── middleware/
│           ├── __init__.py
│           ├── cors.py             # CORS configuration
│           └── logging.py          # Request/response logging
│
├── docs/
│   └── Explanations.md     # Technical documentation
│
└── daily_logs/
    └── day_01.md           # Development logs
```