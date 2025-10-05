import os
import re
import json
import fitz  # PyMuPDF
from tqdm import tqdm

# Paths
RAW_DIR = "../data/raw"
OUTPUT_FILE = "../data/processed/rbi_chunks.json"


# ---------- PDF Extraction ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                text += page_text + " "
    except Exception as e:
        print(f"⚠️ Error extracting {pdf_path}: {e}")
    return normalize_text(text)

def normalize_text(text):
    """Clean and normalize text."""
    text = re.sub(r"\s+", " ", text)   # remove excessive whitespace
    text = re.sub(r"[^a-zA-Z0-9.,;:%()₹\-\s]", "", text)  # remove junk chars but keep finance symbols
    return text.strip()


# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into overlapping chunks (tokens ~ words).
    Helps with context window limits for embeddings/LLMs.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward with overlap

    return chunks


# ---------- Pipeline ----------
def process_pdfs(raw_dir=RAW_DIR, output_file=OUTPUT_FILE):
    all_chunks = []

    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    print(f"📄 Found {len(pdf_files)} PDFs in {raw_dir}")

    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(raw_dir, pdf)
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"⚠️ No text extracted from {pdf}, skipping.")
            continue

        chunks = chunk_text(text, chunk_size=500, overlap=50)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "pdf": pdf,
                "chunk_id": f"{pdf}_{i}",
                "text": chunk
            })

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Extracted & chunked {len(all_chunks)} passages")
    print(f"📂 Saved at: {output_file}")


if __name__ == "__main__":
    process_pdfs()