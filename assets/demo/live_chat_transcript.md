# Live chat transcript

Captured from a running rag-modular-2023 instance on 2026-07-16. The API served
a streaming answer over the HTTP endpoint, grounded in an indexed document,
using a fully local stack: Ollama nomic-embed-text for embeddings, llama3.2:3b
for generation, Postgres 16 with pgvector for hybrid dense plus sparse
retrieval fused with RRF in SQL, and Postgres for memory. The cross encoder
reranker was disabled for this capture (USE_RERANKER=false). No paid keys.

**Indexed document:** assets/demo/rag-modular-2023_overview.txt

**Request:** POST /v1/chat

```json
{"question": "How does rag-modular-2023 do retrieval, and can it run locally?", "model": "llama3.2:3b"}
```

**Streamed answer (verbatim):**

Rag-modular-2023 performs retrieval using a hybrid approach that combines two techniques: dense vector search on PostgreSQL's pgvector and sparse Postgres full-text search.

1. Dense vector search on pgvector is used for efficient embedding lookup.
2. Sparse Postgres full-text search is used to efficiently search within large document collections.

These two ranked lists are then fused using reciprocal rank fusion inside a single SQL query, without requiring index rebuild for each query.

Yes, Rag-modular-2023 can run fully locally and freely, using the same system that runs in production. It uses Ollama models such as nomic-embed-text for embeddings and a local chat model for generation.

**Stream close event:**

```json
{"done": true, "session_id": "5649c3d9-16fa-4a56-954c-946ff0d12a7b", "model": "llama3.2:3b"}
```
