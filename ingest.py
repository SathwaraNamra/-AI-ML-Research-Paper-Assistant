"""
ingest.py
Reads PDFs from the data/ folder, splits them into chunks,
generates embeddings locally, and stores everything in Supabase (pgvector).
"""

import os
import pdfplumber
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DATA_FOLDER = "data"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks to preserve context

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Loading embedding model (first run downloads ~90MB, be patient)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, free, local
print("Model loaded.\n")


# ---------- Helper functions ----------
def extract_text_from_pdf(filepath):
    """Extract text page by page from a PDF, keeping page numbers."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({"page_num": i + 1, "text": text})
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def process_pdf(filepath, filename):
    """Extract, chunk, embed, and return records ready for Supabase insert."""
    records = []
    pages = extract_text_from_pdf(filepath)

    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            records.append({
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "page": page["page_num"]
                }
            })
    return records


# ---------- Main ingestion loop ----------
def main():
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"Create a '{DATA_FOLDER}' folder and add your PDFs first.")

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{DATA_FOLDER}'. Add some papers first.")

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    total_chunks = 0

    for filename in pdf_files:
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"Processing: {filename}")

        records = process_pdf(filepath, filename)

        if not records:
            print(f"  Warning: no extractable text found in {filename}, skipping.\n")
            continue

        # Generate embeddings in batch (faster than one by one)
        texts = [r["content"] for r in records]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        # Attach embeddings and prepare for insert
        rows = []
        for record, embedding in zip(records, embeddings):
            rows.append({
                "content": record["content"],
                "metadata": record["metadata"],
                "embedding": embedding.tolist()
            })

        # Insert into Supabase in batches of 50
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            supabase.table("documents").insert(batch).execute()

        total_chunks += len(rows)
        print(f"  -> {len(rows)} chunks inserted.\n")

    print(f"Done. Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    main()
