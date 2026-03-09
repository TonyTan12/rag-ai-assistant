"""
Document Ingestion Pipeline for RAG System

Supports: PDF, TXT, MD, DOCX files
Processes: Extraction → Chunking → Embedding → Storage
"""

import os
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Match folder structure: Project root is one level up from this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Data", "raw_docs"))
CHROMA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Data", "chroma_db"))

COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "docs")

# Chunking configuration
CHUNK_CHAR_LEN = int(os.getenv("RAG_CHUNK_SIZE", "2500"))
CHUNK_CHAR_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))

# Embedding model
EMBED_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")


@dataclass
class DocChunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


def read_pdf(path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF file.
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of dicts with 'text' and 'page' keys
    """
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = " ".join(text.split())  # Normalize whitespace
            if text.strip():  # Only add non-empty pages
                pages.append({"text": text, "page": i + 1})
        return pages
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return []


def read_txt(path: str) -> List[Dict[str, Any]]:
    """
    Extract text from TXT or MD file.
    
    Args:
        path: Path to text file
        
    Returns:
        List of dicts with 'text' and 'page' keys
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = " ".join(f.read().split())  # Normalize whitespace
        return [{"text": text, "page": 1}] if text.strip() else []
    except Exception as e:
        print(f"Error reading text file {path}: {e}")
        return []


def read_docx(path: str) -> List[Dict[str, Any]]:
    """
    Extract text from DOCX file.
    
    Args:
        path: Path to DOCX file
        
    Returns:
        List of dicts with 'text' and 'page' keys
    """
    try:
        # Try to import python-docx
        try:
            from docx import Document
        except ImportError:
            print(f"python-docx not installed. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "python-docx"])
            from docx import Document
        
        doc = Document(path)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        
        # Join all paragraphs with spaces
        full_text = " ".join(paragraphs)
        full_text = " ".join(full_text.split())  # Normalize whitespace
        
        return [{"text": full_text, "page": 1}] if full_text.strip() else []
    except Exception as e:
        print(f"Error reading DOCX {path}: {e}")
        return []


def chunk_text(text: str, chunk_len: int, overlap: int) -> Iterable[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_len: Characters per chunk
        overlap: Overlap between chunks
        
    Yields:
        Text chunks
    """
    if not text:
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_len, n)
        yield text[start:end]
        if end == n:
            break
        start = max(0, end - overlap)


def stable_chunk_id(source: str, page: int, chunk_index: int, chunk_text_: str) -> str:
    """
    Generate stable unique ID for a chunk.
    
    Args:
        source: Source filename
        page: Page number
        chunk_index: Chunk index on page
        chunk_text_: Chunk text content
        
    Returns:
        Stable hash ID
    """
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(str(page).encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(chunk_text_.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def get_file_reader(ext: str):
    """
    Get appropriate reader function for file extension.
    
    Args:
        ext: File extension (lowercase, with dot)
        
    Returns:
        Reader function or None
    """
    readers = {
        ".pdf": read_pdf,
        ".txt": read_txt,
        ".md": read_txt,
        ".docx": read_docx,
    }
    return readers.get(ext.lower())


def collect_chunks(data_dir: Optional[str] = None) -> List[DocChunk]:
    """
    Collect and chunk all documents in the data directory.
    
    Args:
        data_dir: Directory containing documents (default: DATA_DIR)
        
    Returns:
        List of document chunks
    """
    data_dir = data_dir or DATA_DIR
    chunks: List[DocChunk] = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    # Get all files (not directories)
    files = [f for f in glob.glob(os.path.join(data_dir, "*")) if os.path.isfile(f)]
    
    print(f"Found {len(files)} files in {data_dir}")
    
    processed = 0
    skipped = 0
    
    for path in files:
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()
        
        reader = get_file_reader(ext)
        if not reader:
            print(f"  ⚠ Skipping unsupported file: {filename}")
            skipped += 1
            continue
        
        print(f"  📄 Processing: {filename}")
        pages = reader(path)
        
        if not pages:
            print(f"     ⚠ No content extracted")
            skipped += 1
            continue
        
        file_chunks = 0
        for page_obj in pages:
            page_text = page_obj["text"]
            page_num = page_obj["page"]
            for ci, ctext in enumerate(chunk_text(page_text, CHUNK_CHAR_LEN, CHUNK_CHAR_OVERLAP)):
                cid = stable_chunk_id(filename, page_num, ci, ctext)
                chunks.append(
                    DocChunk(
                        chunk_id=cid,
                        text=ctext,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "chunk_index": ci,
                            "file_type": ext.lstrip(".")
                        },
                    )
                )
                file_chunks += 1
        
        print(f"     ✓ Extracted {file_chunks} chunks from {len(pages)} page(s)")
        processed += 1
    
    print(f"\n📊 Summary: {processed} files processed, {skipped} skipped, {len(chunks)} total chunks")
    return chunks


def ingest_chunks(chunks: List[DocChunk], chroma_dir: Optional[str] = None) -> int:
    """
    Ingest chunks into ChromaDB.
    
    Args:
        chunks: List of document chunks
        chroma_dir: ChromaDB directory (default: CHROMA_DIR)
        
    Returns:
        Number of chunks ingested
    """
    chroma_dir = chroma_dir or CHROMA_DIR
    os.makedirs(chroma_dir, exist_ok=True)

    client = OpenAI()
    chroma_client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    BATCH = 64
    ids, docs, metas = [], [], []
    ingested = 0

    def flush():
        nonlocal ingested
        if not ids:
            return
        print(f"  Embedding batch of {len(ids)} chunks...")
        emb = client.embeddings.create(model=EMBED_MODEL, input=docs)
        vectors = [d.embedding for d in emb.data]
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        ingested += len(ids)
        ids.clear()
        docs.clear()
        metas.clear()

    for ch in chunks:
        ids.append(ch.chunk_id)
        docs.append(ch.text)
        metas.append(ch.metadata)
        if len(ids) >= BATCH:
            flush()

    flush()
    return ingested


def main():
    """Main ingestion entry point."""
    print("=" * 60)
    print("📚 RAG Document Ingestion Pipeline")
    print("=" * 60)
    print(f"\n📁 Data directory: {DATA_DIR}")
    print(f"💾 ChromaDB directory: {CHROMA_DIR}")
    print(f"🔤 Embedding model: {EMBED_MODEL}")
    print(f"📏 Chunk size: {CHUNK_CHAR_LEN} chars (overlap: {CHUNK_CHAR_OVERLAP})")
    print()
    
    try:
        # Collect chunks
        chunks = collect_chunks()
        if not chunks:
            print("❌ No documents found to ingest.")
            return
        
        # Ingest into ChromaDB
        print(f"\n🚀 Ingesting into ChromaDB...")
        ingested = ingest_chunks(chunks)
        
        print(f"\n✅ Success! Ingested {ingested} chunks.")
        print(f"💾 Database location: {CHROMA_DIR}")
        print(f"📚 Collection name: {COLLECTION_NAME}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Please create the directory or check the path.")
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
