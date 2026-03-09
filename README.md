# RAG System - Retrieval Augmented Generation

A production-ready RAG system with hybrid retrieval (vector + BM25), FastAPI backend, and comprehensive document ingestion support.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG SYSTEM ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────┐
│   Documents  │────▶│   Ingestion  │────▶│      ChromaDB Vector Store       │
│  (PDF/TXT/   │     │   Pipeline   │     │  ┌────────────────────────────┐  │
│   DOCX)      │     │              │     │  │  - Document chunks         │  │
└──────────────┘     │  - Chunking  │     │  │  - OpenAI embeddings       │  │
                     │  - Embedding │     │  │  - Metadata (source/page)  │  │
                     │  - Storage   │     │  └────────────────────────────┘  │
                     └──────────────┘     └──────────────────────────────────┘
                                                   │
                           ┌───────────────────────┴───────────────────────┐
                           ▼                                               ▼
                    ┌──────────────┐                              ┌──────────────┐
                    │   Vector     │                              │    BM25      │
                    │  Retrieval   │                              │   Retrieval  │
                    │  (Semantic)  │                              │  (Keyword)   │
                    └──────────────┘                              └──────────────┘
                           │                                               │
                           └───────────────────────┬───────────────────────┘
                                                   ▼
                                        ┌──────────────────┐
                                        │  Hybrid Fusion   │
                                        │  (RRF/Weighted)  │
                                        └──────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────┐
                                        │   OpenAI LLM     │
                                        │  (GPT-4-mini)    │
                                        │                  │
                                        │  - Generate      │
                                        │  - Cite sources  │
                                        └──────────────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────┐
                                        │   FastAPI        │
                                        │   Backend        │
                                        │                  │
                                        │  POST /query     │
                                        │  GET  /health    │
                                        └──────────────────┘
```

## 📁 Project Structure

```
OpenAI Retrieval Augmented Generation/
├── 📂 Rag-Assistant/               # Main application code
│   ├── api.py                      # FastAPI REST API
│   ├── rag.py                      # Core RAG engine with hybrid retrieval
│   ├── ingest.py                   # Document ingestion pipeline
│   ├── app.py                      # Streamlit UI (optional)
│   ├── requirements.txt            # Python dependencies
│   ├── .env                        # Environment variables
│   ├── run_api.ps1                 # API startup script
│   └── test_query.py               # API testing script
│
├── 📂 Data/                        # Data storage
│   ├── 📂 raw_docs/                # Source documents
│   │   ├── about_project.txt
│   │   └── test.txt
│   └── 📂 chroma_db/               # Vector database storage
│       └── chroma.sqlite3
│
├── 📂 Eval/                        # Evaluation and testing
├── 📂 .venv/                       # Python virtual environment
├── 📄 OpenAI Embeddings.docx       # Documentation
└── 📄 README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- Windows PowerShell (for `.ps1` scripts)

### 1. Clone and Setup

```bash
# Navigate to project directory
cd "OpenAI Retrieval Augmented Generation"

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r Rag-Assistant/requirements.txt
```

### 2. Configure Environment

Create a `.env` file in `Rag-Assistant/` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# RAG Configuration
RAG_CHROMA_PATH=../Data/chroma_db
RAG_COLLECTION_NAME=docs
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_GENERATION_MODEL=gpt-4.1-mini

# Hybrid Retrieval Settings
RAG_HYBRID_MODE=weighted        # Options: weighted, rrf, blend
RAG_WEIGHT_VECTOR=0.60          # Vector search weight
RAG_WEIGHT_BM25=0.40            # BM25 keyword weight
RAG_OVERLAP_BONUS=0.10          # Bonus for results in both methods

# Performance
RAG_CANDIDATE_MULTIPLIER=6      # Candidates to retrieve per method
RAG_MAX_CONTEXT_CHARS=12000     # Max context for LLM
```

### 3. Ingest Documents

Place your documents in `Data/raw_docs/`:
- PDF files (`.pdf`)
- Text files (`.txt`, `.md`)
- Word documents (`.docx`)

Run ingestion:

```bash
cd Rag-Assistant
python ingest.py
```

### 4. Start the API Server

```powershell
# Using PowerShell script
.\run_api.ps1

# Or manually
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## 📡 API Documentation

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "ok"
}
```

### Query Endpoint

```bash
POST /query
Content-Type: application/json

{
  "query": "What is the RAG architecture?",
  "k": 6,
  "debug": false
}
```

Response:
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) combines vector search with large language models to provide grounded, citation-based answers [about_project.txt p1 c0].",
  "citations": [
    {
      "source": "about_project.txt",
      "page": 1,
      "chunk_index": 0
    }
  ]
}
```

### Debug Mode

Enable `debug: true` to see retrieval details:

```json
{
  "query": "What is RAG?",
  "k": 4,
  "debug": true
}
```

Response includes:
- `retrieved`: Full chunks with scores
- `debug.timings_ms`: Performance metrics
- `debug.top_scored`: Scoring breakdown

### Debug Endpoints

```bash
# Check environment
GET /debug/env

# Check embedding dimensions
GET /debug/embed_dim?q=hello

# Check ChromaDB status
GET /debug/chroma?q=hello&n=2

# Check available Chroma paths
GET /debug/chroma_paths
```

## 🔧 Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_CHROMA_PATH` | `Data/chroma_db` | Vector database location |
| `RAG_COLLECTION_NAME` | `docs` | ChromaDB collection name |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `RAG_GENERATION_MODEL` | `gpt-4.1-mini` | OpenAI chat model |
| `RAG_HYBRID_MODE` | `weighted` | Fusion: `weighted`, `rrf`, `blend` |
| `RAG_WEIGHT_VECTOR` | `0.60` | Vector search weight |
| `RAG_WEIGHT_BM25` | `0.40` | BM25 keyword weight |
| `RAG_CANDIDATE_MULTIPLIER` | `6` | Candidates per retrieval method |
| `RAG_MAX_CONTEXT_CHARS` | `12000` | Maximum context for LLM |

## 🧪 Testing

### Using the Test Script

```bash
cd Rag-Assistant
python test_query.py
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this project about?", "k": 4}'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is RAG?", "k": 6, "debug": True}
)
print(response.json())
```

## 📊 How It Works

### 1. Document Ingestion

```python
# ingest.py pipeline
1. Read documents (PDF/TXT/DOCX)
2. Extract text and metadata (page numbers)
3. Chunk text (2500 chars, 250 overlap)
4. Generate OpenAI embeddings
5. Store in ChromaDB with metadata
```

### 2. Hybrid Retrieval

```python
# rag.py pipeline
1. Embed query using OpenAI
2. Vector retrieval (semantic similarity)
3. BM25 retrieval (keyword matching)
4. Fuse results (weighted/RRF/blend)
5. Return top-k chunks
```

### 3. Answer Generation

```python
# System prompt rules:
- Answer ONLY using provided context
- Every fact must have citation [source pX cY]
- If no relevant info: "I don't know based on the provided documents."
- No external knowledge allowed
```

## 🔍 Retrieval Modes

### Weighted (Default)
```
score = 0.6 * normalized_vector_score + 0.4 * normalized_bm25_score + overlap_bonus
```

### RRF (Reciprocal Rank Fusion)
```
score = 1/(k + rank_vector) + 1/(k + rank_bm25)
```

### Blend (Hybrid)
```
score = 0.5 * weighted_score + 0.5 * rrf_score
```

## 🛠️ Development

### Adding New Document Types

Edit `ingest.py` and add a reader function:

```python
def read_docx(path: str) -> List[Dict[str, Any]]:
    # Implement DOCX reading
    pass
```

Then update `collect_chunks()` to use it.

### Customizing Chunking

```python
# In ingest.py
CHUNK_CHAR_LEN = 2500      # Characters per chunk
CHUNK_CHAR_OVERLAP = 250   # Overlap between chunks
```

### Adding API Endpoints

Edit `api.py`:

```python
@app.post("/custom")
def custom_endpoint(req: CustomRequest):
    # Your logic here
    return {"result": "success"}
```

## 📈 Performance Tips

1. **Indexing**: Run `ingest.py` once, reuse ChromaDB
2. **BM25 Cache**: Automatically refreshed on startup
3. **Batch Size**: Embeddings processed in batches of 64
4. **Top-K**: Use `k=4-8` for most queries
5. **Debug Mode**: Disable in production for better performance

## 🔒 Security

- Never commit `.env` files
- Use environment variables for API keys
- Run API behind reverse proxy in production
- Enable CORS restrictions as needed

## 🐛 Troubleshooting

### "No documents found"
- Check `Data/raw_docs/` exists and has files
- Verify file extensions (.pdf, .txt, .md)

### "ChromaDB collection empty"
- Run `python ingest.py` first
- Check `RAG_CHROMA_PATH` environment variable

### "OpenAI API error"
- Verify `OPENAI_API_KEY` is set
- Check API key has available credits

### Module not found
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

## 📚 Additional Resources

- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

## 📝 License

This project is for educational and personal use.

---

**Built with ❤️ by Tony Tan**

For questions or support, contact: Tonytan1999aol@gmail.com
