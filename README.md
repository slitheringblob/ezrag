# EZRAG

EZRAG is a lightweight Retrieval-Augmented Generation playground powered by Streamlit, Chroma DB, and SentenceTransformers. 
Drop Markdown or text files into `documents/`, fire up the UI, and start querying your knowledge base.

---

## Project Structure

- `main.py` – Streamlit UI that exposes query input, collection refresh, and results display.
- `chroma_impl.py` – VectorStore wrapper responsible for loading documents, chunking, embedding, and talking to Chroma.
- `documents/` – Place `.md`, `.markdown`, or `.txt` sources here; everything inside is eligible for ingestion.
- `chroma_store/` – Created on first run to persist the Chroma collection locally.

---

## Chroma Strategy

### 1. Chunking
Documents are chunked in `chroma_impl.py` using a tiered approach:
1. Split by Markdown headers (`#`-level 1–3) to preserve meaningful sections.
2. If the file lacks headers, fall back to paragraph splitting on blank lines.
3. As a final guard, split into long sentences (>50 characters) to ensure even sparse text is captured.

### 2. Embeddings & Storage
- Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional cosine space).
- Chunks are packaged with metadata (`source`, `type`, `chunk_number`, `file_hash`) and inserted into a persistent Chroma collection (`chroma_store/`).
- Chroma IDs include the file hash so updates to document content automatically invalidate old entries.
- Before inserting, the store inspects Chroma for existing entries that share the same filename and MD5 hash.
- Unchanged files are skipped; modified files trigger a delete + reinsert cycle

### 3. Search
- Queries are encoded with the same model and sent to Chroma’s cosine-based similarity search.
- Results include chunk metadata and raw text, which the Streamlit app surfaces with relevance hints and source details.

---

## Getting Started

1. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your documents**
   - Drop Markdown/text files into the `documents/` folder.

4. **Run the Streamlit interface**
   ```bash
   streamlit run main.py
   ```
