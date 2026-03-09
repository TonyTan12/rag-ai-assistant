# RAG API Documentation

Complete API reference for the Retrieval-Augmented Generation system.

## Base URL

```
Local Development: http://localhost:8000
```

## Authentication

The API currently does not require authentication for local development. For production deployment, add API key authentication as needed.

## Content-Type

All requests and responses use JSON:
```
Content-Type: application/json
```

---

## Endpoints

### 1. Health Check

Check if the API is running and healthy.

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

### 2. Query Documents

Submit a query to retrieve relevant information from the document store.

```http
POST /query
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | The question or query text |
| `k` | integer | No | 6 | Number of chunks to retrieve (1-30) |
| `debug` | boolean | No | false | Include detailed retrieval info |

**Example Request:**
```json
{
  "query": "What is the RAG architecture?",
  "k": 6,
  "debug": false
}
```

**Example Response:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) combines vector search with large language models to provide grounded, citation-based answers [about_project.txt p1 c0]. The system uses ChromaDB for vector storage and OpenAI models for embeddings and generation.",
  "citations": [
    {
      "source": "about_project.txt",
      "page": 1,
      "chunk_index": 0
    },
    {
      "source": "Tony_Tan_Resume.txt",
      "page": 1,
      "chunk_index": 2
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Generated answer with citations |
| `citations` | array | List of source citations used |
| `citations[].source` | string | Source document filename |
| `citations[].page` | integer | Page number in document |
| `citations[].chunk_index` | integer | Chunk index on page |

**Debug Mode Response:**

When `debug: true`, additional fields are included:

```json
{
  "answer": "...",
  "citations": [...],
  "retrieved": [
    {
      "id": "abc123...",
      "text": "Full chunk text...",
      "metadata": {
        "source": "file.txt",
        "page": 1,
        "chunk_index": 0
      },
      "scores": {
        "vector_distance": 0.234,
        "vector_sim": 0.810,
        "bm25_score": 5.432,
        "fused_score": 0.756
      }
    }
  ],
  "debug": {
    "hybrid_mode": "weighted",
    "k": 6,
    "timings_ms": {
      "vector_retrieval_ms": 145.2,
      "bm25_retrieval_ms": 23.4,
      "total_retrieval_ms": 168.6,
      "generation_ms": 892.3
    }
  }
}
```

**Status Codes:**
- `200 OK` - Query successful
- `422 Unprocessable Entity` - Invalid request body
- `500 Internal Server Error` - RAG processing error

---

### 3. Debug Environment

View environment configuration and paths.

```http
GET /debug/env
```

**Response:**
```json
{
  "cwd": "C:\\...\\Rag-Assistant",
  "env": {
    "RAG_CHROMA_PATH": "../Data/chroma_db",
    "RAG_COLLECTION_NAME": "docs"
  },
  "resolved": {
    "RAG_CHROMA_PATH": "C:\\...\\Data\\chroma_db"
  },
  "engine_cfg": {
    "chroma_path": "../Data/chroma_db",
    "collection_name": "docs"
  }
}
```

---

### 4. Debug Embedding Dimensions

Check the embedding model and vector dimensions.

```http
GET /debug/embed_dim?q={query}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | "hello" | Test query to embed |

**Response:**
```json
{
  "model": "text-embedding-3-small",
  "len": 1536,
  "first5": [0.023, -0.015, 0.008, ...]
}
```

---

### 5. Debug ChromaDB Status

Check ChromaDB connection and query a sample.

```http
GET /debug/chroma?q={query}&n={count}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | "hello" | Test query |
| `n` | integer | 2 | Number of results |

**Response:**
```json
{
  "cwd": "C:\\...",
  "env": {
    "RAG_CHROMA_PATH": "../Data/chroma_db",
    "RAG_COLLECTION_NAME": "docs"
  },
  "has_engine_vector": true,
  "has_collection": true,
  "count": 150,
  "peek_ids": ["id1", "id2", ...],
  "query_ids": [["id1", "id2"]],
  "query_distances": [[0.234, 0.312]]
}
```

---

### 6. Debug Chroma Paths

Check available ChromaDB paths.

```http
GET /debug/chroma_paths
```

**Response:**
```json
{
  "paths": [
    {
      "path": "C:\\...\\Data\\chroma_db",
      "exists": true,
      "files": 3
    }
  ]
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error description"
}
```

**Common Status Codes:**

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | OK | Request successful |
| `422` | Validation Error | Invalid request parameters |
| `500` | Internal Server Error | Server processing error |

---

## Code Examples

### Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())  # {"status": "ok"}

# Query documents
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is Tony's experience with Python?",
        "k": 6,
        "debug": False
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
```

### JavaScript / Fetch

```javascript
// Query documents
fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        query: 'What is Tony\'s experience with Python?',
        k: 6,
        debug: false
    })
})
.then(response => response.json())
.then(data => {
    console.log('Answer:', data.answer);
    console.log('Citations:', data.citations);
});
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the RAG architecture?",
    "k": 6,
    "debug": false
  }'

# Debug mode
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the RAG architecture?",
    "k": 4,
    "debug": true
  }'
```

### PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Query
$body = @{
    query = "What is the RAG architecture?"
    k = 6
    debug = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" -Method POST -ContentType "application/json" -Body $body
```

---

## Rate Limiting

Currently, no rate limiting is implemented for local development. For production:

```python
# Add to api.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest):
    ...
```

---

## CORS Configuration

For production deployment with a frontend:

```python
# Add to api.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Swagger UI

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

Alternative ReDoc documentation:

```
http://localhost:8000/redoc
```

---

## OpenAPI Schema

The complete OpenAPI schema is available at:

```
http://localhost:8000/openapi.json
```

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Contact:** Tonytan1999aol@gmail.com
