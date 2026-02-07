# Multi-Agent RAG System

Production-grade Retrieval-Augmented Generation system with LangGraph multi-agent orchestration and Azure AI integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                        │
│                   /query  /ingest  /metrics                 │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    LangGraph Orchestrator                    │
│                                                             │
│   ┌──────────┐    ┌───────────┐    ┌──────────────────┐    │
│   │  Router   │───▶│ Retriever │───▶│   Synthesizer    │    │
│   │  Agent    │    │   Agent   │    │     Agent        │    │
│   └──────────┘    └───────────┘    └──────────────────┘    │
│        │                                     │              │
│        │          ┌─────────────┐            │              │
│        ├─────────▶│Conversation │            │              │
│        │          └─────────────┘   Quality  │              │
│        │          ┌─────────────┐   Check ───┘              │
│        └─────────▶│Clarification│   (retry if failed)       │
│                   └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼──────┐  ┌───────▼───────┐  ┌───────▼──────┐
│  Azure OpenAI  │  │ Azure AI      │  │    Redis     │
│  (GPT-4o +     │  │ Search        │  │  (Memory)    │
│   Embeddings)  │  │ (Vector Index)│  │              │
└────────────────┘  └───────────────┘  └──────────────┘
```

## Key Features

### RAG Pipeline
- **Dual chunking strategies**: Recursive character splitting and semantic chunking (embedding-based breakpoints)
- **Hybrid retrieval**: Combines vector similarity search and BM25 keyword search
- **Reciprocal Rank Fusion (RRF)**: Merges multi-source results for better recall
- **Token-aware context management**: Respects model context window limits
- **Japanese + English**: Full bilingual support for chunking, retrieval, and generation

### Multi-Agent Orchestration (LangGraph)
- **Router Agent**: Intent classification (retrieval / calculation / conversation / clarification)
- **Retriever Agent**: Query decomposition for multi-hop questions, deduplication
- **Synthesizer Agent**: Response generation with self-evaluation quality check
- **Conditional retry loop**: Re-retrieves if quality check fails (max 2 iterations)
- **Typed state management**: Full `TypedDict` state with `add_messages` reducer

### Evaluation Framework
- **LLM-as-judge metrics**: Faithfulness, relevancy, context precision, context recall
- **Batch evaluation runner**: Load datasets, run evaluations, export JSON reports
- **Self-evaluation in pipeline**: Real-time quality checking during inference

### Production Features
- **Azure AI integration**: Azure OpenAI (GPT-4o + embeddings) + Azure AI Search
- **Conversation memory**: Redis-backed sliding window with token budget management
- **Monitoring & callbacks**: Per-request latency, token usage, cost tracking via LangChain callbacks
- **Structured logging**: JSON (production) / colored console (development) via structlog
- **Containerized deployment**: Docker + docker-compose with Redis
- **CI/CD pipeline**: GitHub Actions (lint, test, Docker build)

## Quick Start

### Prerequisites
- Python 3.11+
- Azure OpenAI resource with GPT-4o and text-embedding-3-large deployments
- Azure AI Search resource
- Redis (optional, for persistent memory)

### Setup

```bash
# Clone and install
cd multi-agent-rag-system
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your Azure credentials

# Start Redis (optional)
docker-compose up redis -d

# Run the application
uvicorn src.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

### API Usage

```bash
# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document content here...",
    "source": "document.pdf",
    "chunking_strategy": "recursive"
  }'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "session_id": "user-123"
  }'

# Check metrics
curl http://localhost:8000/metrics
```

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_chunking.py -v
```

## Project Structure

```
src/
├── config.py                 # Pydantic Settings configuration
├── main.py                   # FastAPI application entry point
├── agents/
│   ├── graph.py              # LangGraph workflow definition
│   ├── router.py             # Intent classification agent
│   ├── retriever.py          # Query decomposition + retrieval agent
│   └── synthesizer.py        # Response generation + quality check agent
├── rag/
│   ├── chunking.py           # Recursive & semantic chunking strategies
│   ├── embeddings.py         # Azure OpenAI embedding service with LRU cache
│   ├── indexer.py            # Azure AI Search index management
│   ├── retriever.py          # Hybrid retrieval with RRF re-ranking
│   └── generator.py          # Context-aware response generation
├── memory/
│   └── conversation.py       # Redis-backed conversation memory
├── tools/
│   ├── search.py             # LangChain-compatible search tools
│   └── calculator.py         # Safe math expression evaluator
├── evaluation/
│   ├── metrics.py            # RAG evaluation metrics (LLM-as-judge)
│   └── runner.py             # Batch evaluation pipeline
└── monitoring/
    ├── callbacks.py          # LangChain callbacks for metrics collection
    └── logger.py             # Structured logging setup
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| LLM Framework | LangChain |
| LLM | Azure OpenAI (GPT-4o) |
| Embeddings | Azure OpenAI (text-embedding-3-large) |
| Vector Search | Azure AI Search |
| API Framework | FastAPI |
| Memory Store | Redis |
| Configuration | Pydantic Settings |
| Logging | structlog |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker |

## License

MIT
