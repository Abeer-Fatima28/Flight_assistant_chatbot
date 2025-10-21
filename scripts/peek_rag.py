from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_store import search

if __name__ == "__main__":
    q = "Do UAE passport holders need a visa for Japan?"
    hits = search(q, k=3)
    for h in hits:
        print(f"[{h['rank']}] {h['title']} ({h['score']:.3f}) — {h['path']}")
        print(h['chunk'][:240], "...\n")
