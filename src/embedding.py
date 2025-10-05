import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
CHUNKS_PATH = "../data/processed/rbi_chunks.json"
INDEX_PATH = "../data/vector_index.faiss"
META_PATH = "../data/vector_metadata.json"

# ---------- Load Model ----------
# Small, fast, and free embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_chunks(path: str):
    """Load text chunks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def create_embeddings(chunks):
    """Convert chunks to embeddings."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def build_faiss_index(embeddings):
    """Create a FAISS index from embeddings."""
    dim = embeddings.shape[1]  # vector dimension
    index = faiss.IndexFlatL2(dim)  # L2 distance index
    index.add(embeddings)
    return index

def save_index(index, path):
    """Save FAISS index to file."""
    faiss.write_index(index, path)

def save_metadata(chunks, path):
    """Save metadata mapping for chunks."""
    metadata = [{"id": i, "source": chunk.get("source", ""), "text": chunk["text"]} for i, chunk in enumerate(chunks)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print("📂 Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"💾 Saving index to {INDEX_PATH}")
    save_index(index, INDEX_PATH)

    print(f"💾 Saving metadata to {META_PATH}")
    save_metadata(chunks, META_PATH)

    print("✅ Done! Embeddings stored in FAISS database.")