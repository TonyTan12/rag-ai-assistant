from __future__ import annotations

import os
import time
import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from openai import OpenAI


logger = logging.getLogger("rag")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("RAG_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


@dataclass(frozen=True)
class RagConfig:
    chroma_path: str = os.getenv("RAG_CHROMA_PATH", "Data/chroma_db")
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "docs")

    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    generation_model: str = os.getenv("RAG_GENERATION_MODEL", "gpt-4.1-mini")

    hybrid_mode: str = os.getenv("RAG_HYBRID_MODE", "weighted")  # weighted|rrf|blend
    weight_vector: float = float(os.getenv("RAG_WEIGHT_VECTOR", "0.60"))
    weight_bm25: float = float(os.getenv("RAG_WEIGHT_BM25", "0.40"))
    overlap_bonus: float = float(os.getenv("RAG_OVERLAP_BONUS", "0.10"))

    candidate_multiplier: int = int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "6"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))

    norm_p_low: float = float(os.getenv("RAG_NORM_P_LOW", "5"))
    norm_p_high: float = float(os.getenv("RAG_NORM_P_HIGH", "95"))

    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: Dict[str, Any]

    vector_distance: Optional[float] = None
    vector_sim: Optional[float] = None
    bm25_score: Optional[float] = None

    norm_vector: Optional[float] = None
    norm_bm25: Optional[float] = None

    rank_vector: Optional[int] = None
    rank_bm25: Optional[int] = None

    fused_score: Optional[float] = None

    from_vector: bool = False
    from_bm25: bool = False


def _tokenize(text: str) -> List[str]:
    return (text or "").lower().split()


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _robust_minmax(values: List[float], p_low: float, p_high: float) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    lo = _percentile(values, p_low)
    hi = _percentile(values, p_high)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _scale_01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _distance_to_similarity(d: float) -> float:
    if d is None:
        return 0.0
    if d < 0:
        d = 0.0
    return 1.0 / (1.0 + d)


class VectorRetriever:
    def __init__(self, cfg: RagConfig, client: OpenAI):
        self.cfg = cfg
        self.openai = client
        self._chroma = chromadb.PersistentClient(
            path=self.cfg.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_collection(self.cfg.collection_name)

    def embed_query(self, query: str) -> List[float]:
        emb = self.openai.embeddings.create(model=self.cfg.embedding_model, input=query)
        return emb.data[0].embedding

    def retrieve(self, query: str, n: int) -> List[RetrievedChunk]:
        q_emb = self.embed_query(query)
        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[RetrievedChunk] = []
        for cid, text, md, dist in zip(ids, docs, metas, dists):
            dist_f = float(dist) if dist is not None else None
            out.append(
                RetrievedChunk(
                    id=str(cid),
                    text=str(text),
                    metadata=md or {},
                    vector_distance=dist_f,
                    vector_sim=_distance_to_similarity(dist_f) if dist_f is not None else None,
                    from_vector=True,
                )
            )
        return out


class BM25Retriever:
    def __init__(self, cfg: RagConfig, chroma_collection):
        self.cfg = cfg
        self._collection = chroma_collection
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_metas: List[Dict[str, Any]] = []
        self._doc_tokens: List[List[str]] = []

    def refresh(self) -> None:
        batch = 5000
        offset = 0

        ids_all: List[str] = []
        docs_all: List[str] = []
        metas_all: List[Dict[str, Any]] = []

        while True:
            res = self._collection.get(include=["documents", "metadatas"], limit=batch, offset=offset)
            ids = res.get("ids", [])
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            if not ids:
                break
            ids_all.extend(ids)
            docs_all.extend(docs)
            metas_all.extend(metas)
            offset += len(ids)
            if len(ids) < batch:
                break

        f_ids: List[str] = []
        f_txt: List[str] = []
        f_md: List[Dict[str, Any]] = []
        f_tok: List[List[str]] = []

        for cid, txt, md in zip(ids_all, docs_all, metas_all):
            tokens = _tokenize(txt or "")
            if not tokens:
                continue
            f_ids.append(str(cid))
            f_txt.append(txt or "")
            f_md.append(md or {})
            f_tok.append(tokens)

        self._doc_ids = f_ids
        self._doc_texts = f_txt
        self._doc_metas = f_md
        self._doc_tokens = f_tok

        if not self._doc_ids:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(self._doc_tokens)

    def retrieve(self, query: str, n: int) -> List[RetrievedChunk]:
        if self._bm25 is None:
            self.refresh()
            if self._bm25 is None:
                return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        out: List[RetrievedChunk] = []
        for rank, i in enumerate(idxs, start=1):
            out.append(
                RetrievedChunk(
                    id=self._doc_ids[i],
                    text=self._doc_texts[i],
                    metadata=self._doc_metas[i],
                    bm25_score=float(scores[i]),
                    from_bm25=True,
                    rank_bm25=rank,
                )
            )
        return out


class HybridRetriever:
    def __init__(self, cfg: RagConfig, vector: VectorRetriever, bm25: BM25Retriever):
        self.cfg = cfg
        self.vector = vector
        self.bm25 = bm25

    def retrieve(self, query: str, k: int, debug: bool = False) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
        if k < 1:
            return [], {}

        n_cand = max(k, k * self.cfg.candidate_multiplier)

        t0 = time.perf_counter()
        vec = self.vector.retrieve(query, n=n_cand)
        t_vec_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        lex = self.bm25.retrieve(query, n=n_cand)
        t_lex_ms = (time.perf_counter() - t1) * 1000.0

        by_id: Dict[str, RetrievedChunk] = {}

        for r, ch in enumerate(vec, start=1):
            ch.rank_vector = r
            by_id[ch.id] = ch

        for ch in lex:
            if ch.id in by_id:
                e = by_id[ch.id]
                e.from_bm25 = True
                e.bm25_score = ch.bm25_score
                e.rank_bm25 = ch.rank_bm25
            else:
                by_id[ch.id] = ch

        cands = list(by_id.values())

        if self.cfg.hybrid_mode == "rrf":
            self._apply_rrf(cands)
        elif self.cfg.hybrid_mode == "blend":
            self._apply_weighted(cands)
            weighted = {c.id: (c.fused_score or 0.0) for c in cands}
            self._apply_rrf(cands)
            for c in cands:
                c.fused_score = 0.5 * (c.fused_score or 0.0) + 0.5 * weighted.get(c.id, 0.0)
        else:
            self._apply_weighted(cands)

        cands.sort(key=lambda x: (x.fused_score or 0.0), reverse=True)
        top = cands[:k]

        dbg: Dict[str, Any] = {}
        if debug:
            dbg = {
                "hybrid_mode": self.cfg.hybrid_mode,
                "k": k,
                "n_candidates_requested_each": n_cand,
                "n_union_candidates": len(cands),
                "timings_ms": {
                    "vector_retrieval_ms": t_vec_ms,
                    "bm25_retrieval_ms": t_lex_ms,
                },
                "top_scored": [asdict(x) for x in top],
            }

        return top, dbg

    def _apply_weighted(self, cands: List[RetrievedChunk]) -> None:
        vec_sims = [float(c.vector_sim) for c in cands if c.vector_sim is not None]
        bm25_scores = [float(c.bm25_score) for c in cands if c.bm25_score is not None]

        v_lo, v_hi = _robust_minmax(vec_sims, self.cfg.norm_p_low, self.cfg.norm_p_high)
        b_lo, b_hi = _robust_minmax(bm25_scores, self.cfg.norm_p_low, self.cfg.norm_p_high)

        for c in cands:
            c.norm_vector = _scale_01(float(c.vector_sim), v_lo, v_hi) if c.vector_sim is not None else 0.0
            c.norm_bm25 = _scale_01(float(c.bm25_score), b_lo, b_hi) if c.bm25_score is not None else 0.0
            overlap = 1.0 if (c.from_vector and c.from_bm25) else 0.0
            c.fused_score = (
                self.cfg.weight_vector * (c.norm_vector or 0.0)
                + self.cfg.weight_bm25 * (c.norm_bm25 or 0.0)
                + self.cfg.overlap_bonus * overlap
            )

    def _apply_rrf(self, cands: List[RetrievedChunk]) -> None:
        k_rrf = self.cfg.rrf_k
        for c in cands:
            s = 0.0
            if c.rank_vector is not None:
                s += 1.0 / (k_rrf + float(c.rank_vector))
            if c.rank_bm25 is not None:
                s += 1.0 / (k_rrf + float(c.rank_bm25))
            c.fused_score = s


SYSTEM_RULES = (
  "You are a retrieval-augmented assistant.\n"
  "Answer ONLY using the provided Context.\n"
  "Every factual sentence MUST end with at least one citation label exactly as shown in Context, e.g. [file p1 c0].\n"
  "If Context is empty or does not contain the answer, reply exactly:\n"
  "\"I don't know based on the provided documents.\"\n"
  "Do not use knowledge outside Context. Do not invent citations.\n"
)


def _make_label(md: Dict[str, Any]) -> str:
    src = md.get("source", "unknown")
    page = md.get("page", None)
    ci = md.get("chunk_index", None)
    p = f"p{page}" if page is not None else "p?"
    c = f"c{ci}" if ci is not None else "c?"
    return f"[{src} {p} {c}]"


def build_context(chunks: List[RetrievedChunk], max_chars: int) -> str:
    parts: List[str] = []
    used = 0
    for ch in chunks:
        label = _make_label(ch.metadata)
        block = f"{label}\n{(ch.text or '').strip()}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


class RagEngine:
    def __init__(self, cfg: Optional[RagConfig] = None, openai_client: Optional[OpenAI] = None):
        self.cfg = cfg or RagConfig()
        self.openai = openai_client or OpenAI()
        self.vector = VectorRetriever(self.cfg, self.openai)
        self.bm25 = BM25Retriever(self.cfg, self.vector._collection)
        self.hybrid = HybridRetriever(self.cfg, self.vector, self.bm25)

    def refresh_bm25(self) -> None:
        self.bm25.refresh()

    def answer(self, query: str, k: int = 8, debug: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        top, dbg = self.hybrid.retrieve(query, k=k, debug=debug)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        context = build_context(top, max_chars=self.cfg.max_context_chars)

        t1 = time.perf_counter()
        resp = self.openai.chat.completions.create(
            model=self.cfg.generation_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        gen_ms = (time.perf_counter() - t1) * 1000.0

        answer_text = (resp.choices[0].message.content or "").strip()

        # ---- ADD THIS BLOCK RIGHT HERE ----
        if answer_text != "I don't know based on the provided documents.":
            labels = [_make_label(ch.metadata) for ch in top]
            if not any(lbl in answer_text for lbl in labels):
                answer_text = "I don't know based on the provided documents."
        # ------------------------------------

        out: Dict[str, Any] = {
            "answer": answer_text,
            "citations": [
                {
                    "source": ch.metadata.get("source"),
                    "page": ch.metadata.get("page"),
                    "chunk_index": ch.metadata.get("chunk_index"),
                }
                for ch in top
            ],
        }

        out: Dict[str, Any] = {
            "answer": answer_text,
            "citations": [
                {
                    "source": ch.metadata.get("source"),
                    "page": ch.metadata.get("page"),
                    "chunk_index": ch.metadata.get("chunk_index"),
                }
                for ch in top
            ],
        }

        if debug:
            out["retrieved"] = [
                {
                    "id": ch.id,
                    "text": ch.text,
                    "metadata": ch.metadata,
                    "scores": {
                        "vector_distance": ch.vector_distance,
                        "vector_sim": ch.vector_sim,
                        "bm25_score": ch.bm25_score,
                        "norm_vector": ch.norm_vector,
                        "norm_bm25": ch.norm_bm25,
                        "rank_vector": ch.rank_vector,
                        "rank_bm25": ch.rank_bm25,
                        "fused_score": ch.fused_score,
                        "from_vector": ch.from_vector,
                        "from_bm25": ch.from_bm25,
                    },
                }
                for ch in top
            ]
            out["debug"] = dbg
            out["debug"]["timings_ms"]["total_retrieval_ms"] = retrieval_ms
            out["debug"]["timings_ms"]["generation_ms"] = gen_ms

        return out