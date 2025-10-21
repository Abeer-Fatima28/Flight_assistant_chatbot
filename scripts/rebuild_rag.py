from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_store import build_index

if __name__ == "__main__":
    index, metas, ids = build_index(force_rebuild=True)
    print(f"Built RAG index at data/vectorstore/ with {len(metas)} chunks.")
