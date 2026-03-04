"""
Flask API server for Personal Knowledge Base RAG chatbot.
"""
import os
import io
import json
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context

# PDF / text extraction
import pypdf
from bs4 import BeautifulSoup

from rag_engine import (
    ingest_text, retrieve, compute_confidence, build_context,
    stream_chat, list_sources, delete_source, db_stats, get_db
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".pdf", ".txt", ".md", ".html", ".htm"}


# ─────────────────────────────────────────────────────────────────────────
# Static frontend
# ─────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ─────────────────────────────────────────────────────────────────────────
# Upload & ingest
# ─────────────────────────────────────────────────────────────────────────

def extract_text_from_file(filepath: Path, filename: str) -> str:
    ext = filepath.suffix.lower()
    if ext == ".pdf":
        reader = pypdf.PdfReader(str(filepath))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    elif ext in (".html", ".htm"):
        raw = filepath.read_text(errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    else:  # .txt, .md
        return filepath.read_text(errors="ignore")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    fname = Path(file.filename)
    if fname.suffix.lower() not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {fname.suffix}"}), 400

    save_path = UPLOAD_DIR / fname.name
    file.save(str(save_path))

    try:
        content = extract_text_from_file(save_path, fname.name)
        if not content.strip():
            return jsonify({"error": "File appears empty or unreadable"}), 400

        result = ingest_text(content, source_name=fname.name,
                             source_type=fname.suffix.lstrip("."))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest-text", methods=["POST"])
def ingest_raw_text():
    data = request.get_json()
    if not data or not data.get("content"):
        return jsonify({"error": "No content provided"}), 400

    content = data["content"].strip()
    name    = data.get("name", f"note_{int(time.time())}.txt")
    stype   = data.get("type", "note")

    try:
        result = ingest_text(content, source_name=name, source_type=stype)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest-url", methods=["POST"])
def ingest_url():
    """Fetch a URL, extract text, ingest."""
    import urllib.request
    data = request.get_json()
    url  = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL"}), 400

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string.strip() if soup.title else url

        result = ingest_text(text, source_name=title[:80], source_type="url")
        result["title"] = title
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────
# Query / Chat  (streaming SSE)
# ─────────────────────────────────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        results    = retrieve(question)
        confidence = compute_confidence(results)
        context    = build_context(results)

        sources_info = [
            {
                "source": r["source"],
                "score":  round(r["score"], 4),
                "preview": r["content"][:200] + "…" if len(r["content"]) > 200 else r["content"],
            }
            for r in results
        ]

        def generate():
            # First send metadata
            meta = json.dumps({
                "type":       "meta",
                "confidence": confidence,
                "sources":    sources_info,
            })
            yield f"data: {meta}\n\n"

            # Then stream LLM tokens
            for token in stream_chat(question, context):
                chunk = json.dumps({"type": "token", "text": token})
                yield f"data: {chunk}\n\n"

            # Done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":  "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────
# Knowledge base management
# ─────────────────────────────────────────────────────────────────────────

@app.route("/api/sources", methods=["GET"])
def sources():
    return jsonify(list_sources())


@app.route("/api/sources/<path:source_name>", methods=["DELETE"])
def remove_source(source_name):
    deleted = delete_source(source_name)
    return jsonify({"deleted_chunks": deleted, "source": source_name})


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(db_stats())


@app.route("/api/health", methods=["GET"])
def health():
    """Check Ollama connectivity."""
    import requests as req
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return jsonify({"status": "ok", "ollama": True, "models": models})
    except Exception as e:
        return jsonify({"status": "degraded", "ollama": False, "error": str(e)}), 200


if __name__ == "__main__":
    print("🚀 RAG Knowledge Base starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)