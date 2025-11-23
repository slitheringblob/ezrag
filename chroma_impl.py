import glob
import hashlib
import logging
import os
import re
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Simple local RAG helper backed by Chroma DB with SentenceTransformer embeddings.
    """

    def __init__(
        self,
        documents_dir: str = "documents",
        persist_directory: str = "chroma_store",
        collection_name: str = "ezrag_chroma",
        model_name: str = "all-MiniLM-L6-v2",
        auto_ingest: bool = True,
    ):
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name

        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info("Loading embedding model '%s'...", model_name)
        self.model = SentenceTransformer(model_name)

        logger.info("Connecting to Chroma (persisted at %s)...", self.persist_directory)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        if auto_ingest:
            self.ingest_documents()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_documents(self, force: bool = False) -> None:
        """
        Load markdown/text files, chunk, embed and store in Chroma.
        When `force` is False we only embed files whose content hash changed.
        """
        doc_groups = self._prepare_documents()
        pending_chunks = []

        for group in doc_groups:
            if not group["chunks"]:
                continue

            source = group["source"]
            file_hash = group["file_hash"]
            already_vectorized = self._is_file_vectorized(source, file_hash)

            if already_vectorized and not force:
                logger.info("Skipping '%s' (already vectorized).", source)
                continue

            logger.info("Indexing '%s'...", source)
            self._delete_source(source)
            pending_chunks.extend(group["chunks"])

        if not pending_chunks:
            logger.info("No new or updated documents to ingest.")
            return

        texts = [chunk["content"] for chunk in pending_chunks]
        logger.info("Encoding %d chunks...", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

        self.collection.add(
            ids=[chunk["id"] for chunk in pending_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in pending_chunks],
        )
        logger.info("Inserted %d chunks into '%s'.", len(pending_chunks), self.collection_name)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query.strip():
            return []

        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()

        try:
            response = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        documents = response.get("documents") or []
        if not documents:
            return []

        docs = documents[0]
        metas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results = []
        for doc, meta, distance in zip(docs, metas, distances):
            meta = meta or {}
            results.append(
                {
                    "content": doc,
                    "source": meta.get("source"),
                    "type": meta.get("type"),
                    "chunk_number": meta.get("chunk_number"),
                    "distance": distance,
                }
            )

        return results

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.ingest_documents(force=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_documents(self) -> List[Dict]:
        documents = []
        patterns = ("*.md", "*.markdown", "*.txt")
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.documents_dir, pattern)))
        files = sorted(set(files))

        if not files:
            logger.warning("No documents found in '%s'.", self.documents_dir)
            return []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
            except (OSError, UnicodeDecodeError) as exc:
                logger.error("Failed to read '%s': %s", file_path, exc)
                continue

            file_hash = self._hash_content(content)
            filename = os.path.basename(file_path)
            doc_type = os.path.splitext(filename)[0]
            chunks = self._split_markdown_into_chunks(content)

            chunk_entries = []
            for index, chunk in enumerate(chunks, start=1):
                text = chunk.strip()
                if not text:
                    continue

                chunk_entries.append(
                    {
                        "id": f"{file_hash}-{index}",
                        "content": text,
                        "metadata": {
                            "source": filename,
                            "type": doc_type,
                            "chunk_number": index,
                            "file_hash": file_hash,
                        },
                    }
                )

            documents.append(
                {
                    "source": filename,
                    "file_hash": file_hash,
                    "chunks": chunk_entries,
                }
            )

        return documents

    def _split_markdown_into_chunks(self, content: str) -> List[str]:
        """
        Chunk markdown first by headers, then by paragraphs/sentences as fallback.
        """
        header_pattern = r"(?=\n#{1,3}\s+)"
        sections = re.split(header_pattern, content)
        chunks = [section.strip() for section in sections if section.strip()]

        if len(chunks) <= 1:
            chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

        if len(chunks) <= 1:
            sentences = re.split(r"[.!?]+", content)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 50]

        return chunks

    def _hash_content(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _is_file_vectorized(self, source: str, file_hash: str) -> bool:
        try:
            result = self.collection.get(
                where={"source": source},
                include=["metadatas"],
                limit=1,
            )
        except Exception:
            return False

        metadatas = result.get("metadatas") or []
        if not metadatas:
            return False

        stored_hash = metadatas[0].get("file_hash")
        return stored_hash == file_hash

    def _delete_source(self, source: str) -> None:
        try:
            self.collection.delete(where={"source": source})
        except Exception:
            pass
