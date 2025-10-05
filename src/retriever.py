import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
INDEX_PATH = "../data/vector_index.faiss"
META_PATH = "../data/vector_metadata.json"

# ---------- Load Model ----------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------- Load FAISS + Metadata ----------
def load_index(index_path=INDEX_PATH, meta_path=META_PATH):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ---------- Query Function ----------
def search(query, index, metadata, top_k=5):
    """Retrieve top_k most similar chunks for a query."""
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec, dtype=np.float32), top_k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:  # no match
            continue
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "text": metadata[idx]["text"],
            "source": metadata[idx].get("source", f"chunk_{idx}")
        })
    return results

# ---------- Example Run ----------
if __name__ == "__main__":
    print("📂 Loading FAISS index...")
    index, metadata = load_index()

    query = input("🔍 Enter your query: ")

    print(f"\n🔎 Searching for: {query}\n")
    results = search(query, index, metadata, top_k=5)

    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"Text: {r['text'][:300]}...\n")  # preview first 300 chars