# Multi-Agent RAG System

Production-grade Retrieval-Augmented Generation system with LangGraph multi-agent orchestration and Azure AI integration. Built with Python 3.11, LangChain/LangGraph, Azure OpenAI, and Azure AI Search.

## Architecture

```
                            ┌──────────────┐
                            │   Client     │
                            └──────┬───────┘
                                   │  HTTP
                    ┌──────────────▼──────────────┐
                    │       FastAPI Gateway        │
                    │  /query  /ingest  /metrics   │
                    │  /health /session/{id}       │
                    └──────────────┬───────────────┘
                                   │
         ┌─────────────────────────▼─────────────────────────┐
         │              LangGraph StateGraph                  │
         │                                                    │
         │  ┌────────┐   intent    ┌──────────────────────┐  │
         │  │ Router ├────────────▶│   Conditional Edge   │  │
         │  │ Agent  │  classify   │                      │  │
         │  └────────┘             │  retrieval ──┐       │  │
         │                         │  calculation─┤       │  │
         │                         │  conversation│       │  │
         │                         │  clarification       │  │
         │                         └──────┬───────┘       │  │
         │           ┌────────────────────┤               │  │
         │           │                    │               │  │
         │  ┌────────▼─────┐    ┌────────▼────────┐      │  │
         │  │  Retriever   │    │  Conversation   │      │  │
         │  │  Agent       │    │  Node           │      │  │
         │  │ ┌──────────┐ │    └────────┬────────┘      │  │
         │  │ │  Query   │ │             │               │  │
         │  │ │  Decomp  │ │             ▼ END           │  │
         │  │ └──────────┘ │                              │  │
         │  └──────┬───────┘                              │  │
         │         │                                      │  │
         │  ┌──────▼────────┐                             │  │
         │  │  Synthesizer  │    quality_check             │  │
         │  │  Agent        ├───── failed? ──▶ retry ─┐   │  │
         │  │ ┌───────────┐ │     (max 2x)            │   │  │
         │  │ │  Quality  │ │◀────────────────────────┘   │  │
         │  │ │  Check    │ │                             │  │
         │  │ └───────────┘ │                             │  │
         │  └──────┬────────┘                             │  │
         │         ▼ END                                  │  │
         └────────────────────────────────────────────────┘  │
                                                              │
           ┌──────────────────┬───────────────┬───────────────┘
           │                  │               │
  ┌────────▼───────┐  ┌──────▼──────┐  ┌─────▼──────┐
  │  Azure OpenAI  │  │ Azure AI    │  │   Redis    │
  │  GPT-4o        │  │ Search      │  │  Mem Store  │
  │  Embeddings    │  │ Vector+BM25 │  │  (TTL)     │
  └────────────────┘  └─────────────┘  └────────────┘
```

## Why This Architecture

| Design Decision | Rationale |
|---|---|
| **LangGraph StateGraph** over simple chains | Enables conditional routing, retry loops, and typed shared state — essential for production multi-agent workflows |
| **Hybrid Retrieval (Vector + BM25)** with RRF | Vector search captures semantic similarity; BM25 catches exact keyword matches; RRF fusion gives better recall than either alone |
| **Query Decomposition** in Retriever Agent | Complex multi-hop questions ("Compare A's policy with B's approach") need to be broken into sub-queries for accurate retrieval |
| **Self-Evaluation Loop** in Synthesizer | Catches hallucination and irrelevance before returning to the user; automatically retries with fresh retrieval if quality check fails |
| **Redis-backed Memory** with token budget | Sliding window prevents unbounded growth; token budget ensures context window is never exceeded; Redis survives restarts |
| **Pydantic Settings** for config | Type-safe configuration with environment variable binding — no string parsing errors in production |

## Key Features

### RAG Pipeline (`src/rag/`)

**Chunking** — Two strategies, selectable per document:
- `RecursiveChunker`: Hierarchical splitting (paragraph → sentence → word → character) with configurable token overlap. Supports Japanese sentence boundaries (`。`, `、`).
- `SemanticChunker`: Embedding-based breakpoint detection — groups semantically similar consecutive sentences. Splits when cosine distance exceeds threshold or token budget is reached.

**Retrieval** — Hybrid search with Reciprocal Rank Fusion:
```
Vector Search (cosine similarity) ──┐
                                     ├──▶ RRF Merge ──▶ Top-N Results
BM25 Keyword Search ────────────────┘
```
RRF score = `Σ 1/(k + rank_i)` across result sets, where `k=60`. Documents appearing in both sets get boosted scores.

**Generation** — Token-aware context injection:
- Builds context string from ranked documents, respecting `max_context_tokens` budget
- Supports bilingual (Japanese/English) — auto-detects and matches the user's language
- Includes source citations in responses

### Multi-Agent Orchestration (`src/agents/`)

The system uses a **LangGraph `StateGraph`** with typed `AgentState`:

```python
class AgentState(TypedDict):
    query: str                              # User's question
    intent: str                             # Classified intent
    messages: Annotated[list, add_messages]  # Conversation history (reducer)
    documents: list[RetrievedDocument]       # Retrieved context
    answer: str                             # Generated response
    sources: list[str]                      # Citation sources
    quality_check: dict[str, object]        # Self-eval result
    iteration: int                          # Retry counter
    error: str | None                       # Error tracking
```

**Agents:**

| Agent | Role | Key Logic |
|---|---|---|
| `RouterAgent` | Intent classification | Zero-temperature LLM call → one of `retrieval` / `calculation` / `conversation` / `clarification` |
| `RetrieverAgent` | Query decomposition + retrieval | Breaks complex questions into 1-3 sub-queries, retrieves per sub-query, deduplicates by `(source, chunk_index)` |
| `SynthesizerAgent` | Generation + quality check | Generates answer with citations, then self-evaluates on `faithful` / `relevant` / `complete`. If faithfulness or relevancy fails → triggers retry loop (max 2 iterations) |

**Graph Flow:**
```
START → route → [retrieve → synthesize → (retry?)] → END
              → conversation → END
              → clarification → END
```

### Evaluation Framework (`src/evaluation/`)

**Four LLM-as-judge metrics** (weighted overall score):

| Metric | Weight | What It Measures |
|---|---|---|
| Faithfulness | 30% | Is every claim in the answer supported by the context? |
| Relevancy | 30% | Does the answer address the question? |
| Context Precision | 20% | What fraction of retrieved docs are actually relevant? |
| Context Recall | 20% | Does the context cover all info needed to answer? |

**Batch evaluation runner** — Load datasets from JSON, run evaluations, export reports:
```json
{
  "samples": [
    {
      "question": "RAGの仕組みを説明してください",
      "ground_truth": "RAGは検索と生成を組み合わせた手法で...",
      "contexts": ["..."],
      "answer": "..."
    }
  ]
}
```

### Monitoring (`src/monitoring/`)

**LangChain callback handler** (`MonitoringCallback`) attached to every request:
- Per-step latency tracking (ms)
- Input/output token counts
- Estimated cost calculation (Azure OpenAI pricing)
- Error capture per step

**Metrics aggregator** — Exposes via `/metrics` endpoint:
```json
{
  "total_requests": 1523,
  "avg_latency_ms": 2340.5,
  "p95_latency_ms": 4120.0,
  "avg_cost_usd": 0.0032,
  "total_cost_usd": 4.8756,
  "error_counts": {"synthesize": 3}
}
```

### Conversation Memory (`src/memory/`)

```
┌─────────────────────────────────────────┐
│          ConversationMemory             │
│                                         │
│  Backend: Redis (prod) / dict (dev)     │
│  Sliding window: max_turns × 2 msgs    │
│  Token budget: most recent first        │
│  TTL: 3600s per session (configurable)  │
└─────────────────────────────────────────┘
```

- `add_message(role, content)` — Appends + auto-trims
- `get_history()` — Returns messages within token budget, most-recent-first priority
- `clear()` — Wipes session
- `get_summary()` — Returns `{message_count, total_tokens, ...}`

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Process a question through the multi-agent pipeline |
| `POST` | `/ingest` | Ingest document text into the vector store |
| `GET` | `/metrics` | Aggregated performance metrics |
| `GET` | `/health` | Service health check |
| `DELETE` | `/session/{id}` | Clear conversation memory for a session |

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAGシステムのアーキテクチャを説明してください",
    "session_id": "user-abc-123"
  }'
```

Response:
```json
{
  "answer": "RAGシステムは以下の主要コンポーネントで構成されています...[Source: architecture.pdf]",
  "sources": ["architecture.pdf"],
  "session_id": "user-abc-123",
  "quality_check": {
    "faithful": true,
    "relevant": true,
    "complete": true,
    "suggestion": null
  },
  "metrics": {
    "request_id": "req-uuid",
    "total_latency_ms": 2450.32,
    "total_tokens": 1823,
    "estimated_cost_usd": 0.003215,
    "steps": [
      {"name": "route", "latency_ms": 320.1, "tokens": 150, "error": null},
      {"name": "retrieve", "latency_ms": 890.5, "tokens": 423, "error": null},
      {"name": "synthesize", "latency_ms": 1240.7, "tokens": 1250, "error": null}
    ]
  }
}
```

### POST /ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ドキュメント全文テキスト...",
    "source": "company_policy.pdf",
    "chunking_strategy": "recursive"
  }'
```

Response:
```json
{
  "source": "company_policy.pdf",
  "chunks_indexed": 42
}
```

## Quick Start

### Prerequisites
- Python 3.11+
- Azure OpenAI resource (GPT-4o + text-embedding-3-large deployments)
- Azure AI Search resource
- Redis (optional — falls back to in-memory for development)

### Local Development

```bash
cd multi-agent-rag-system

# Install with dev dependencies
pip install -e ".[dev]"

# Configure Azure credentials
cp .env.example .env
# Edit .env:
#   AZURE_OPENAI_API_KEY=your-key
#   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
#   AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
#   AZURE_SEARCH_API_KEY=your-key

# (Optional) Start Redis for persistent memory
docker-compose up redis -d

# Run the service
uvicorn src.main:app --reload --port 8000
```

### Docker Deployment

```bash
# Build and start all services (app + Redis)
docker-compose up --build

# Or build image separately
docker build -t multi-agent-rag:latest .
```

## Configuration

All settings are managed via environment variables through Pydantic Settings:

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | Azure OpenAI API version |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-large` | Embedding deployment name |
| `AZURE_SEARCH_ENDPOINT` | — | Azure AI Search endpoint |
| `AZURE_SEARCH_API_KEY` | — | Azure AI Search admin key |
| `AZURE_SEARCH_INDEX_NAME` | `documents` | Search index name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `TOP_K` | `5` | Retrieval candidates per search |
| `RERANK_TOP_N` | `3` | Final documents after RRF re-ranking |
| `MAX_CONTEXT_TOKENS` | `4096` | Max tokens for context injection |
| `ENVIRONMENT` | `development` | `development` or `production` (affects logging format) |

## Running Tests

```bash
# All 43 tests
pytest

# With verbose output and coverage
pytest -v --cov=src --cov-report=term-missing

# Specific test modules
pytest tests/test_chunking.py -v    # Chunking strategies
pytest tests/test_retriever.py -v   # RRF merge logic
pytest tests/test_agents.py -v      # Router agent (mocked LLM)
pytest tests/test_memory.py -v      # Conversation memory
pytest tests/test_monitoring.py -v  # Metrics tracking
pytest tests/test_calculator.py -v  # Tool safety
pytest tests/test_evaluation.py -v  # Eval metric weights
```

Test design — all unit tests run **without Azure credentials or network access**:
- **Agent tests**: Mock `AzureOpenAI` client, verify intent classification logic
- **Retriever tests**: Test RRF merge algorithm in isolation (pure math, no API calls)
- **Chunking tests**: Splitting, overlap, metadata, Japanese text support
- **Memory tests**: In-memory backend, sliding window, token budget
- **Monitoring tests**: Latency calculation, cost estimation, aggregation
- **Calculator tests**: Safe AST-based math expression evaluation

## Project Structure

```
multi-agent-rag-system/
├── pyproject.toml                # Dependencies, build config, tool settings
├── Dockerfile                    # Multi-stage production image
├── docker-compose.yml            # App + Redis orchestration
├── .github/workflows/ci.yml     # CI pipeline (lint → test → Docker build)
│
├── src/
│   ├── config.py                 # Pydantic Settings (env var binding)
│   ├── tokenizer.py              # Token counting (tiktoken w/ graceful fallback)
│   ├── main.py                   # FastAPI app, endpoints, lifespan
│   │
│   ├── agents/                   # LangGraph multi-agent system
│   │   ├── graph.py              #   StateGraph definition, nodes, edges
│   │   ├── router.py             #   Intent classification agent
│   │   ├── retriever.py          #   Query decomposition + retrieval
│   │   └── synthesizer.py        #   Response generation + self-evaluation
│   │
│   ├── rag/                      # RAG pipeline components
│   │   ├── chunking.py           #   RecursiveChunker + SemanticChunker
│   │   ├── embeddings.py         #   Azure OpenAI embeddings + LRU cache
│   │   ├── indexer.py            #   Azure AI Search index management
│   │   ├── retriever.py          #   Hybrid retrieval + RRF re-ranking
│   │   └── generator.py          #   Context-aware response generation
│   │
│   ├── memory/
│   │   └── conversation.py       #   Redis / in-memory conversation history
│   │
│   ├── tools/                    # LangChain-compatible tools
│   │   ├── search.py             #   Document search tool
│   │   └── calculator.py         #   Safe math expression evaluator (AST-based)
│   │
│   ├── evaluation/               # Offline evaluation framework
│   │   ├── metrics.py            #   LLM-as-judge (4 metrics)
│   │   └── runner.py             #   Batch eval + JSON report export
│   │
│   └── monitoring/               # Observability
│       ├── callbacks.py          #   LangChain callbacks (latency/token/cost)
│       └── logger.py             #   structlog config (JSON prod / color dev)
│
└── tests/                        # 43 unit tests (no network required)
    ├── test_agents.py
    ├── test_calculator.py
    ├── test_chunking.py
    ├── test_evaluation.py
    ├── test_memory.py
    ├── test_monitoring.py
    └── test_retriever.py
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent Orchestration | LangGraph `StateGraph` | Typed state, conditional edges, retry loops |
| LLM Framework | LangChain Core | Tool abstraction, callback protocol |
| Chat Model | Azure OpenAI GPT-4o | Intent classification, generation, evaluation |
| Embeddings | Azure OpenAI text-embedding-3-large | Document + query vectorization |
| Vector Store | Azure AI Search (HNSW) | Vector similarity + BM25 keyword search |
| API Layer | FastAPI + Pydantic | Typed endpoints, auto-generated OpenAPI docs |
| Memory | Redis 7 | Session-scoped conversation persistence |
| Config | Pydantic Settings | Type-safe env var binding |
| Logging | structlog | Structured JSON (prod) / colored console (dev) |
| Token Counting | tiktoken (w/ fallback) | Context window management |
| Testing | pytest + pytest-cov | 43 tests, offline-capable |
| Linting | Ruff | Fast Python linter + formatter |
| Type Check | mypy (strict mode) | Static type analysis |
| CI/CD | GitHub Actions | Lint → Test → Docker build |
| Container | Docker + docker-compose | Production deployment |

## License

MIT
