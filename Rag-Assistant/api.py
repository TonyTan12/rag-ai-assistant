from __future__ import annotations

from dotenv import load_dotenv
from narwhals import col
load_dotenv()

import os
from pathlib import Path
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple

from rag import RagEngine

app = FastAPI(title="RAG API", version="0.3")

engine = RagEngine()

@app.get("/debug/env")
def debug_env():
    cwd = os.getcwd()
    chroma_env = os.getenv("RAG_CHROMA_PATH")
    collection_env = os.getenv("RAG_COLLECTION_NAME")
    resolved_chroma = str(Path(chroma_env).expanduser().resolve()) if chroma_env else None
    cfg = getattr(engine, "cfg", None)

    return {
        "cwd": cwd,
        "env": {
            "RAG_CHROMA_PATH": chroma_env,
            "RAG_COLLECTION_NAME": collection_env,
        },
        "resolved": {"RAG_CHROMA_PATH": resolved_chroma},
        "engine_cfg": {
            "chroma_path": getattr(cfg, "chroma_path", None),
            "collection_name": getattr(cfg, "collection_name", None),
        },
    }

@app.get("/debug/embed_dim")
def debug_embed_dim(q: str = "hello"):
    try:
        v = engine.vector
        emb = v.embed_query(q)
        return {
            "model": engine.cfg.embedding_model,
            "len": len(emb),
            "first5": emb[:5],
        }
    except Exception as e:
        return {"error": repr(e), "traceback": traceback.format_exc()}

@app.get("/debug/chroma_paths")
def debug_chroma_paths():
    base = r"D:\tonyt\OpenAI Retrieval Augmented Generation"
    paths = [
        os.path.join(base, "Data", "chroma_db"),
        os.path.join(base, "Rag-Assistant", "Data", "chroma_db"),
    ]
    out = {"paths": []}
    for p in paths:
        out["paths"].append({
            "path": p,
            "exists": os.path.exists(p),
            "files": (len(os.listdir(p)) if os.path.exists(p) else 0),
        })
    return out


@app.get("/debug/chroma")
def debug_chroma(q: str = "hello", n: int = 2):
    try:
        out = {
            "cwd": os.getcwd(),
            "env": {
                "RAG_CHROMA_PATH": os.getenv("RAG_CHROMA_PATH"),
                "RAG_COLLECTION_NAME": os.getenv("RAG_COLLECTION_NAME"),
            },
        }

        vr = getattr(engine, "vector", None)
        out["has_engine_vector"] = vr is not None
        out["engine_vector_type"] = str(type(vr)) if vr is not None else None

        # Try multiple attribute names to locate the collection
        candidates = ["_collection", "collection", "col"]
        col = None
        for name in candidates:
            if vr is not None and hasattr(vr, name):
                col = getattr(vr, name)
                out["collection_attr"] = name
                break

        out["has_collection"] = col is not None
        out["collection_type"] = str(type(col)) if col is not None else None

        if col is None:
            return out

        out["count"] = col.count()

        got = col.get(limit=min(max(n, 1), 10))
        out["peek_ids"] = got.get("ids", [])
        out["peek_n_ids"] = len(out["peek_ids"])

        q_emb = engine.vector.embed_query(q)
        res = col.query(query_embeddings=[q_emb], n_results=min(max(n, 1), 10))
        out["query_ids"] = res.get("ids", [])
        out["query_distances"] = res.get("distances", [])
        return out

    except Exception as e:
        return {"error": repr(e), "traceback": traceback.format_exc()}


@app.on_event("startup")
def _startup() -> None:
    # Avoid first-request latency spike from lazy BM25 build
    try:
        engine.refresh_bm25()
    except Exception:
        # Don’t fail startup if BM25 refresh has an issue; vector retrieval can still work.
        pass


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=30)
    debug: bool = False


class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    citations: Optional[List[Citation]] = None
    retrieved: Optional[List[RetrievedChunk]] = None
    # optional: surface retrieval debug if you want (kept out of schema for strictness)


@app.get("/health")
def health():
    return {"status": "ok"}


def _normalize_retrieved(raw: Any) -> List[RetrievedChunk]:
    if not raw:
        return []

    out: List[RetrievedChunk] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cid = str(item[0])
                text = str(item[1])
                meta = item[2] if len(item) >= 3 and isinstance(item[2], dict) else {}
                out.append(RetrievedChunk(id=cid, text=text, metadata=meta))
            elif isinstance(item, dict):
                out.append(
                    RetrievedChunk(
                        id=str(item.get("id", "")),
                        text=str(item.get("text", "")),
                        metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    )
                )
            else:
                out.append(RetrievedChunk(id="", text=str(item), metadata={}))
        return out

    return [RetrievedChunk(id="", text=str(raw), metadata={})]


def _dedupe_citations(citations: List[Citation]) -> List[Citation]:
    seen: set[Tuple[str, Optional[int], Optional[int]]] = set()
    out: List[Citation] = []
    for c in citations:
        key = (c.source, c.page, c.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _filter_citations_used_in_answer(answer: str, citations: List[Citation]) -> List[Citation]:
    used: List[Citation] = []
    for c in citations:
        if c.page is None or c.chunk_index is None:
            continue
        label = f"[{c.source} p{c.page} c{c.chunk_index}]"
        if label in answer:
            used.append(c)
    return used


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result: Dict[str, Any] = engine.answer(req.query, k=req.k, debug=req.debug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {type(e).__name__}") from e

    answer = (result.get("answer") or "").strip()

    # Citations: prefer engine-provided citations (aligned to retrieved top-k)
    citations_raw = result.get("citations") or []
    citations_all = _dedupe_citations(
        [
            Citation(
                source=str(c.get("source")),
                page=c.get("page"),
                chunk_index=c.get("chunk_index"),
            )
            for c in citations_raw
            if isinstance(c, dict) and c.get("source")
        ]
    )
    citations_used = _filter_citations_used_in_answer(answer, citations_all)

    if req.debug:
        retrieved = _normalize_retrieved(result.get("retrieved"))
        return QueryResponse(answer=answer, citations=citations_used, retrieved=retrieved)

    return QueryResponse(answer=answer, citations=citations_used)