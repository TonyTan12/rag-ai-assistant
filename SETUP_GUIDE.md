# Setup Guide & Usage Examples

Complete guide for setting up and using the RAG system.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Git** (optional) - [Download](https://git-scm.com/downloads)
- **Windows PowerShell** (for `.ps1` scripts)

### Required Accounts
- **OpenAI API Key** - [Get one here](https://platform.openai.com/api-keys)
  - New accounts get $5 in free credits
  - Costs approximately $0.0001 per 1K tokens for embeddings

### System Requirements
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space
- **Internet:** Required for OpenAI API calls

---

## Installation

### Step 1: Navigate to Project Directory

```powershell
cd "C:\Users\Julia\Desktop\OpenAI Retrieval Augmented Generation"
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# You should see (.venv) in your prompt
```

### Step 3: Install Dependencies

```powershell
# Install all required packages
pip install -r Rag-Assistant\requirements.txt
```

Expected output:
```
Successfully installed chromadb-1.5.1 fastapi-0.131.0 openai-1.75.0 ...
```

### Step 4: Verify Installation

```powershell
# Check Python version
python --version  # Should be 3.10+

# Check key packages
python -c "import chromadb; import openai; import fastapi; print('All packages installed!')"
```

---

## Configuration

### Step 1: Create Environment File

Create a file at `Rag-Assistant\.env`:

```powershell
# Navigate to Rag-Assistant directory
cd Rag-Assistant

# Create .env file
notepad .env
```

### Step 2: Add Configuration

Paste this into the `.env` file:

```env
# ============================================
# OpenAI Configuration (REQUIRED)
# ============================================
OPENAI_API_KEY=sk-your-openai-api-key-here

# ============================================
# RAG System Configuration
# ============================================
# Vector database path (relative to api.py)
RAG_CHROMA_PATH=../Data/chroma_db

# Collection name in ChromaDB
RAG_COLLECTION_NAME=docs

# Embedding model (OpenAI)
RAG_EMBEDDING_MODEL=text-embedding-3-small

# Generation model (OpenAI)
RAG_GENERATION_MODEL=gpt-4.1-mini

# ============================================
# Hybrid Retrieval Settings
# ============================================
# Mode: weighted | rrf | blend
RAG_HYBRID_MODE=weighted

# Weights for weighted mode (must sum to ~1.0)
RAG_WEIGHT_VECTOR=0.60
RAG_WEIGHT_BM25=0.40

# Bonus for results in both methods
RAG_OVERLAP_BONUS=0.10

# ============================================
# Performance Settings
# ============================================
# Multiplier for candidate retrieval
RAG_CANDIDATE_MULTIPLIER=6

# RRF constant (for rrf mode)
RAG_RRF_K=60

# Max characters to send to LLM
RAG_MAX_CONTEXT_CHARS=12000

# ============================================
# Logging
# ============================================
RAG_LOG_LEVEL=INFO
```

### Step 3: Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. Paste it in your `.env` file

⚠️ **Important:** Never share your API key or commit it to version control!

---

## Running the System

### Option 1: Using PowerShell Script (Recommended)

```powershell
# Make sure you're in the Rag-Assistant directory
cd "C:\Users\Julia\Desktop\OpenAI Retrieval Augmented Generation\Rag-Assistant"

# Activate virtual environment
..\.venv\Scripts\activate

# Run the API
.\run_api.ps1
```

### Option 2: Using Uvicorn Directly

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Navigate to Rag-Assistant
cd Rag-Assistant

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Using Python

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Navigate to Rag-Assistant
cd Rag-Assistant

# Start the server
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Verify Server is Running

Open your browser and go to:
- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

You should see:
```json
{"status": "ok"}
```

---

## Usage Examples

### Example 1: Ingest Documents

Before querying, you need to ingest documents:

```powershell
# Make sure virtual environment is activated
.venv\Scripts\activate

# Navigate to Rag-Assistant
cd Rag-Assistant

# Run ingestion
python ingest.py
```

Expected output:
```
============================================================
📚 RAG Document Ingestion Pipeline
============================================================

📁 Data directory: C:\...\Data\raw_docs
💾 ChromaDB directory: C:\...\Data\chroma_db
🔤 Embedding model: text-embedding-3-small
📏 Chunk size: 2500 chars (overlap: 250)

Found 3 files in C:\...\Data\raw_docs
  📄 Processing: Tony_Tan_Resume.txt
     ✓ Extracted 12 chunks from 1 page(s)
  📄 Processing: about_project.txt
     ✓ Extracted 8 chunks from 1 page(s)

📊 Summary: 2 files processed, 1 skipped, 20 total chunks

🚀 Ingesting into ChromaDB...
  Embedding batch of 20 chunks...

✅ Success! Ingested 20 chunks.
💾 Database location: C:\...\Data\chroma_db
📚 Collection name: docs
```

### Example 2: Basic Query (cURL)

```powershell
# Simple query
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "What is Tony\'s work experience?", "k": 4}'
```

Response:
```json
{
  "answer": "Tony Tan has experience at Ernst & Young U.S. LLP in Tax Technology & Transformation. He is currently a Senior (July 2025 - Present) and previously worked as Staff (January 2023 - July 2025) [Tony_Tan_Resume.txt p1 c0].",
  "citations": [
    {"source": "Tony_Tan_Resume.txt", "page": 1, "chunk_index": 0}
  ]
}
```

### Example 3: Query with Debug Mode

```powershell
# Query with detailed debug info
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "What programming languages does Tony know?", "k": 4, "debug": true}'
```

### Example 4: Python Script

Create a file `test_api.py`:

```python
import requests
import json

API_URL = "http://localhost:8000"

def query_documents(question, k=6, debug=False):
    """Query the RAG system."""
    response = requests.post(
        f"{API_URL}/query",
        json={"query": question, "k": k, "debug": debug}
    )
    return response.json()

def main():
    # Example questions
    questions = [
        "What is Tony's educational background?",
        "What technical skills does Tony have?",
        "What projects has Tony worked on?",
        "What is the RAG architecture?"
    ]
    
    print("=" * 60)
    print("RAG System Query Examples")
    print("=" * 60)
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        print("-" * 60)
        
        result = query_documents(q, k=4)
        
        print(f"💡 Answer: {result['answer']}")
        print(f"\n📚 Citations:")
        for citation in result.get('citations', []):
            print(f"   - {citation['source']} (page {citation['page']})")
        print("=" * 60)

if __name__ == "__main__":
    main()
```

Run it:
```powershell
python test_api.py
```

### Example 5: PowerShell Script

```powershell
$apiUrl = "http://localhost:8000"

# Health check
Write-Host "Checking health..." -ForegroundColor Cyan
$health = Invoke-RestMethod -Uri "$apiUrl/health"
Write-Host "Status: $($health.status)" -ForegroundColor Green

# Query
$body = @{
    query = "What is Tony's experience with Python?"
    k = 4
    debug = $false
} | ConvertTo-Json

Write-Host "`nQuerying..." -ForegroundColor Cyan
$response = Invoke-RestMethod -Uri "$apiUrl/query" -Method POST -ContentType "application/json" -Body $body

Write-Host "`nAnswer: $($response.answer)" -ForegroundColor Green
Write-Host "`nCitations:" -ForegroundColor Yellow
$response.citations | ForEach-Object {
    Write-Host "  - $($_.source) (page $($_.page))"
}
```

### Example 6: Adding Custom Documents

1. Copy your documents to `Data/raw_docs/`:
   - PDF files (`.pdf`)
   - Text files (`.txt`, `.md`)
   - Word documents (`.docx`)

2. Re-run ingestion:
   ```powershell
   cd Rag-Assistant
   python ingest.py
   ```

3. Query your documents:
   ```powershell
   curl -X POST http://localhost:8000/query `
     -H "Content-Type: application/json" `
     -d '{"query": "Your question here", "k": 6}'
   ```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```powershell
# Make sure virtual environment is activated
.venv\Scripts\activate

# Reinstall dependencies
pip install -r Rag-Assistant\requirements.txt
```

### Issue: "No documents found"

**Solution:**
```powershell
# Check if Data/raw_docs exists
Test-Path "Data\raw_docs"

# List files in the directory
Get-ChildItem "Data\raw_docs"

# Check file extensions (must be .pdf, .txt, .md, or .docx)
```

### Issue: "OpenAI API error"

**Solution:**
```powershell
# Check if .env file exists
Test-Path "Rag-Assistant\.env"

# Verify API key format (should start with sk-)
Get-Content "Rag-Assistant\.env" | Select-String "OPENAI_API_KEY"

# Test API key
python -c "import openai; openai.api_key='your-key'; print(openai.Model.list())"
```

### Issue: "ChromaDB collection empty"

**Solution:**
```powershell
# Run ingestion first
cd Rag-Assistant
python ingest.py

# Check if ChromaDB was created
Get-ChildItem "..\Data\chroma_db"
```

### Issue: "Port 8000 already in use"

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Or use a different port
uvicorn api:app --host 0.0.0.0 --port 8001
```

### Issue: "Permission denied" on PowerShell

**Solution:**
```powershell
# Run PowerShell as Administrator
# Or set execution policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Advanced Configuration

### Custom Chunking

Edit `ingest.py`:
```python
CHUNK_CHAR_LEN = 3000      # Larger chunks
CHUNK_CHAR_OVERLAP = 300   # More overlap
```

### Different Embedding Model

Edit `.env`:
```env
RAG_EMBEDDING_MODEL=text-embedding-3-large  # Better quality, more expensive
```

### Different Generation Model

Edit `.env`:
```env
RAG_GENERATION_MODEL=gpt-4o  # More capable, more expensive
```

### Pure Vector Search (No BM25)

Edit `.env`:
```env
RAG_HYBRID_MODE=weighted
RAG_WEIGHT_VECTOR=1.0
RAG_WEIGHT_BM25=0.0
```

---

## Next Steps

1. **Explore the API:** Visit http://localhost:8000/docs
2. **Add more documents:** Copy files to `Data/raw_docs/`
3. **Customize prompts:** Edit `SYSTEM_RULES` in `rag.py`
4. **Build a frontend:** Connect to the API from your website
5. **Deploy:** Use Docker or cloud services for production

---

**Need Help?**
- Email: Tonytan1999aol@gmail.com
- Check the main README.md for architecture details
- Review API_DOCUMENTATION.md for endpoint details
