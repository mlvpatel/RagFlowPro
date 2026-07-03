# Live chat transcript

Captured from a running RagFlowPro instance on 2026-07-03. The API served a
streaming answer over the HTTP endpoint, grounded in an indexed document, using
a fully local stack (Ollama nomic-embed-text for embeddings, llama3.2:3b for
generation, pgvector for hybrid retrieval, Postgres for memory). No paid keys.

**Indexed document:** assets/demo/ragflowpro_overview.txt

**Request:** POST /v1/chat

```json
{"question": "How does RagFlowPro do retrieval, and can it run locally?", "model": "llama3.2:3b"}
```

**Streamed answer:**

According to the context provided, RagFlowPro's retrieval method is hybrid, combining:

1. Dense vector search on pgvector.
2. Sparse Postgres full text search.

These two ranked lists are then fused using reciprocal rank fusion inside a single SQL query.

Regarding running locally, it seems that RagFlowPro can run fully locally and for free. This is achieved using:

1. Ollama models (specifically nomic-embed-text) for producing embeddings.
2. A local chat model for generation.

These components enable RagFlowPro to operate independently of cloud based services, allowing for self hosting or deployment on premises.

**Result:** the answer is correct and grounded in the source document. The
response streamed token by token as server sent events, and the turn was
persisted to Postgres under a session id for conversational memory.
