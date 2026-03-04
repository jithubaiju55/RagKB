"""
RAG Engine: Embedding, Vector Storage, Retrieval with Confidence Scoring
Uses Ollama mxbai-embed-large for embeddings and llama3 for generation.
"""

import os
import json
import re
import sqlite3
import hashlib
import time
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- Config ---
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large:latest"
CHAT_MODEL = "llama3:latest"
DB_PATH = "db/knowledge_base.db"
CHUNK_SIZE = 400       # tokens ~approx chars/4
CHUNK_OVERLAP = 80
TOP_K = 5

# ── Confidence thresholds ──────────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.72   # cosine sim
CONFIDENCE_MEDIUM = 0.50
# below MEDIUM = Low


# ═══════════════════════════════════════════════════════════════════════════
# Vector DB  (SQLite + numpy cosine similarity — no extra deps)
# ═══════════════════════════════════════════════════════════════════════════

def get_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            content     TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            doc_hash    TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON documents(doc_hash)")
    conn.commit()
    return conn


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════
# Ollama helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> np.ndarray:
    """Call Ollama embed endpoint. Returns float32 numpy array."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"embeddings": [[...]] }
    vec = data.get("embeddings") or data.get("embedding")
    if isinstance(vec[0], list):
        vec = vec[0]
    return np.array(vec, dtype=np.float32)


def get_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
    """Batch embed via Ollama for speed."""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    vecs = data.get("embeddings") or data.get("embedding")
    return [np.array(v, dtype=np.float32) for v in vecs]


def stream_chat(prompt: str, context: str):
    """Stream LLM response. Yields text chunks."""
    system = (
        "You are a precise knowledge assistant. "
        "Answer ONLY using the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Be concise and cite which source you used when possible."
    )
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
    
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": full_prompt},
            ],
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 512},
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue


# ═══════════════════════════════════════════════════════════════════════════
# Text chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by character count (fast)."""
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size * 4   # ~4 chars per token
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap * 4
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Ingest pipeline
# ═══════════════════════════════════════════════════════════════════════════

def file_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def ingest_text(content: str, source_name: str, source_type: str = "text") -> Dict:
    """Chunk, embed (batch), store. Returns stats."""
    conn = get_db()
    doc_h = file_hash(content)

    # Skip if already ingested
    existing = conn.execute(
        "SELECT COUNT(*) as c FROM documents WHERE doc_hash=?", (doc_h,)
    ).fetchone()["c"]
    if existing > 0:
        conn.close()
        return {"status": "duplicate", "chunks": existing, "source": source_name}

    chunks = chunk_text(content)
    if not chunks:
        conn.close()
        return {"status": "empty", "chunks": 0, "source": source_name}

    # Batch embed all chunks
    t0 = time.time()
    embeddings = get_embeddings_batch(chunks)
    embed_time = time.time() - t0

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        rows.append((source_name, source_type, chunk, i, doc_h,
                     emb.tobytes(), time.time()))

    conn.executemany(
        "INSERT INTO documents (source_name, source_type, content, chunk_index, doc_hash, embedding, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "chunks": len(chunks),
        "source": source_name,
        "embed_time": round(embed_time, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval + confidence
# ═══════════════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """Embed query, cosine-search all stored vectors, return top_k with scores."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, source_name, source_type, content, embedding FROM documents"
    ).fetchall()
    conn.close()

    if not rows:
        return []

    q_emb = get_embedding(query)
    scored = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        sim = cosine_similarity(q_emb, emb)
        scored.append({
            "id": row["id"],
            "source": row["source_name"],
            "type": row["source_type"],
            "content": row["content"],
            "score": sim,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def compute_confidence(results: List[Dict]) -> Dict:
    """Derive a confidence level from top retrieval scores."""
    if not results:
        return {"level": "none", "score": 0.0, "label": "No Sources", "pct": 0}

    top_score = results[0]["score"]
    avg_score = np.mean([r["score"] for r in results[:3]])
    combined  = (top_score * 0.7 + avg_score * 0.3)

    if combined >= CONFIDENCE_HIGH:
        level, label, color = "high",   "High Confidence",   "#22c55e"
    elif combined >= CONFIDENCE_MEDIUM:
        level, label, color = "medium", "Medium Confidence", "#f59e0b"
    else:
        level, label, color = "low",    "Low Confidence",    "#ef4444"

    return {
        "level":  level,
        "label":  label,
        "color":  color,
        "score":  round(float(combined), 4),
        "top":    round(float(top_score), 4),
        "pct":    int(combined * 100),
    }


def build_context(results: List[Dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Source {i}: {r['source']} | relevance {r['score']:.2f}]\n{r['content']}")
    return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# DB utilities
# ═══════════════════════════════════════════════════════════════════════════

def list_sources() -> List[Dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT source_name, source_type, COUNT(*) as chunks, MAX(created_at) as last_added
        FROM documents GROUP BY source_name
        ORDER BY last_added DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_source(source_name: str) -> int:
    conn = get_db()
    cur = conn.execute("DELETE FROM documents WHERE source_name=?", (source_name,))
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted


def db_stats() -> Dict:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()["c"]
    sources = conn.execute("SELECT COUNT(DISTINCT source_name) as c FROM documents").fetchone()["c"]
    conn.close()
    return {"total_chunks": total, "total_sources": sources}