import os, json, glob
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

INDEX_DIR   = os.getenv("RAG_INDEX_DIR", "data/vectorstore")
DOC_GLOB_RAW = os.getenv("RAG_DOC_GLOB", "data/**/*.md;data/**/*.txt")
EMB_MODEL   = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE  = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_K       = int(os.getenv("RAG_TOP_K", "5"))

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _read_text(fpath: str) -> str:
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fpath, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += step
    return chunks

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def _split_globs(spec: str) -> List[str]:
    spec = spec.replace(",", ";")
    return [p.strip() for p in spec.split(";") if p.strip()]

def _collect_files() -> List[str]:
    patterns = _split_globs(DOC_GLOB_RAW)
    paths = []
    for pat in patterns:
        paths.extend([p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)])
    # de-dup while preserving order
    seen, dedup = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); dedup.append(p)
    return dedup

def _paths() -> Dict[str, str]:
    return {
        "index": os.path.join(INDEX_DIR, "index.faiss"),
        "metas": os.path.join(INDEX_DIR, "metas.jsonl"),
        "ids":   os.path.join(INDEX_DIR, "ids.txt"),
    }

def _save_index(index, metas: List[Dict[str, Any]], ids: List[str]):
    _ensure_dir(INDEX_DIR)
    p = _paths()
    faiss.write_index(index, p["index"])
    with open(p["metas"], "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(p["ids"], "w", encoding="utf-8") as f:
        f.write("\n".join(ids))

def _load_index() -> Tuple[faiss.Index, List[Dict[str, Any]], List[str]]:
    p = _paths()
    index = faiss.read_index(p["index"])
    metas, ids = [], []
    with open(p["metas"], "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    with open(p["ids"], "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return index, metas, ids

def index_exists() -> bool:
    p = _paths()
    return os.path.exists(p["index"]) and os.path.exists(p["metas"]) and os.path.exists(p["ids"])

def build_index(force_rebuild: bool = False) -> Tuple[faiss.Index, List[Dict[str, Any]], List[str]]:
    if index_exists() and not force_rebuild:
        return _load_index()

    model = SentenceTransformer(EMB_MODEL)
    paths = _collect_files()
    if not paths:
        print(f"[RAG] No documents matched RAG_DOC_GLOB='{DOC_GLOB_RAW}'. "
              f"Try setting it to 'data/**/*.md' or similar.")
        # build empty index
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        _save_index(index, [], [])
        return index, [], []

    docs = []
    for pth in paths:
        txt = _read_text(pth)
        if not txt.strip():
            continue
        title = os.path.basename(pth)
        chunks = _chunk_text(txt, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            docs.append({
                "id": f"{pth}#chunk_{i}",
                "title": title,
                "path": pth,
                "chunk": ch.strip()
            })

    if not docs:
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        _save_index(index, [], [])
        return index, [], []

    texts = [d["chunk"] for d in docs]
    emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    emb = _normalize_rows(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact cosine via inner product on normalized vectors
    index.add(emb)

    _save_index(index, docs, [d["id"] for d in docs])
    print(f"[RAG] Indexed {len(docs)} chunks from {len(paths)} files → {INDEX_DIR}")
    return index, docs, [d["id"] for d in docs]

def search(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if not index_exists():
        build_index(force_rebuild=False)

    index, metas, ids = _load_index()
    model = SentenceTransformer(EMB_MODEL)

    q = model.encode([query], normalize_embeddings=False)
    q = np.asarray(q, dtype="float32")
    q = _normalize_rows(q)

    scores, idxs = index.search(q, k)  # shapes: (1,k)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    out = []
    for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(metas):
            continue
        m = metas[i]
        out.append({
            "rank": rank,
            "score": float(sc),
            "title": m.get("title"),
            "path": m.get("path"),
            "chunk": m.get("chunk"),
            "id": m.get("id")
        })
    return out
