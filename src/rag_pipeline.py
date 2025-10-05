import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ---------- Paths ----------
INDEX_PATH = "../data/vector_index.faiss"
META_PATH = "../data/vector_metadata.json"

# ---------- Load Models ----------
print("🧠 Loading embedding model (MiniLM)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("🤖 Loading free LLM model (Flan-T5)...")
LLM_MODEL = "google/flan-t5-base"  # change to larger one if you have GPU
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

# ---------- Load FAISS + Metadata ----------
def load_index(index_path=INDEX_PATH, meta_path=META_PATH):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ---------- Retriever ----------
def retrieve_context(query, index, metadata, top_k=5):
    """Return top_k most relevant chunks for a query."""
    query_vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_vec, dtype=np.float32), top_k)

    retrieved = []
    for idx in I[0]:
        if idx == -1:
            continue
        retrieved.append(metadata[idx]["text"])
    return " ".join(retrieved)


# ---------- Generator ----------
def generate_answer(query, context, max_input_tokens=350):
    """Generate a final answer using the LLM with safely truncated context."""
    # Split long context
    words = context.split()
    if len(words) > max_input_tokens:
        context = " ".join(words[:max_input_tokens])

    # Build compact prompt
    prompt = f"""
You are an assistant that answers based only on the given context.
Context: {context}

Question: {query}

Answer briefly and clearly:
""".strip()

    output = llm(prompt)[0]["generated_text"]
    return output




# ---------- RAG Pipeline ----------
def rag_query(query, index, metadata, top_k=5):
    """Full RAG pipeline: retrieve → generate answer."""
    print("\n🔍 Retrieving relevant context...")
    context = retrieve_context(query, index, metadata, top_k=top_k)

    if not context.strip():
        return "No relevant context found."

    print("💡 Generating answer from context...")
    answer = generate_answer(query, context)
    return answer


# ---------- Run Example ----------
if __name__ == "__main__":
    print("📂 Loading FAISS index & metadata...")
    index, metadata = load_index()

    while True:
        query = input("\n🧭 Enter your query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = rag_query(query, index, metadata, top_k=5)

        print("\n🧾 Final Answer:\n", answer)
        print("\n" + "=" * 80)
