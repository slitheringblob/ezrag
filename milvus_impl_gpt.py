# milvus_impl.py

import os
import re
import glob
import logging
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Milvus Lite VectorStore for local file-backed embeddings.
    """

    def __init__(
        self,
        path: str = "./ezrag_milvus_lite.db",
        collection_name: str = "ezrag_collection",
        dim: int = 384,
        auto_insert=False
    ):
        self.path = path
        self.collection_name = collection_name
        self.dim = dim

        logger.info("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("Connecting Milvus Lite at: %s", path)
        self.client = MilvusClient(uri=path)

        # Create or load collection, INCLUDING INDEX
        self._make_local_collection()

        # Prepare documents/chunks
        self.documents = self._prepare_documents()

        if auto_insert and self.documents:
            self._insert_data_if_needed()

    # ----------------------------------------------------------------------
    # Create Collection with Index (Corrected)
    # ----------------------------------------------------------------------
    def _make_local_collection(self):
        """
        Creates the collection if missing.
        ALWAYS creates IVF_FLAT index immediately.
        """

        exists = False
        try:
            resp = self.client.has_collection(self.collection_name)
            exists = resp["has_collection"] if isinstance(resp, dict) else bool(resp)
        except Exception:
            exists = False

        if not exists:
            logger.info("Creating new collection: %s", self.collection_name)

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="ezrag vector store",
                enable_dynamic_field=True,
            )

            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                dimension=self.dim,
            )
            
            try:
                self.client.create_index(
                    collection_name=self.collection_name,
                    field_name="vector",
                    index_params={
                        "index_type": "FLAT",
                        "metric_type": "L2",
                        "params": {}
                    }
                )
            except Exception as e:
                print("Index may already exist:", e)


        else:
            logger.info("Collection '%s' already exists", self.collection_name)

        # Final: load collection into memory
        self.client.load_collection(self.collection_name)

    # ----------------------------------------------------------------------
    # Read markdown & chunk
    # ----------------------------------------------------------------------
    def _prepare_documents(self):
        documents = []
        all_chunks = []

        documents_dir = "documents"
        os.makedirs(documents_dir, exist_ok=True)

        md_files = glob.glob(os.path.join(documents_dir, "*.md"))
        logger.info("Found %d markdown files", len(md_files))

        for file_path in md_files:
            filename = os.path.basename(file_path)

            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    content = fh.read()

                doc_type = filename.rsplit(".", 1)[0]
                chunks = self._split_markdown_into_chunks(content)

                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        all_chunks.append({
                            "content": chunk.strip(),
                            "source": filename,
                            "type": doc_type,
                            "chunk_number": i + 1,
                        })

            except Exception as e:
                logger.exception("Error reading %s: %s", filename, e)

        if not all_chunks:
            logger.info("No document chunks found.")
            return []

        # Embed chunks
        texts = [c["content"] for c in all_chunks]
        embeddings = self.model.encode(texts).tolist()

        # Build documents
        for chunk, emb in zip(all_chunks, embeddings):
            documents.append({
                "vector": emb,
                "content": chunk["content"],
                "source": chunk["source"],
                "type": chunk["type"],
                "chunk_number": chunk["chunk_number"],
            })

        logger.info("Prepared %d total chunks", len(documents))
        return documents

    # ----------------------------------------------------------------------
    # Chunking
    # ----------------------------------------------------------------------
    def _split_markdown_into_chunks(self, content: str):
        header_pattern = r"(?=\n#{1,3}\s+)"
        sections = re.split(header_pattern, content)
        chunks = [s.strip() for s in sections if s.strip()]

        if len(chunks) <= 1:
            chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

        if len(chunks) <= 1:
            sentences = re.split(r"[.!?]+", content)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 50]

        return chunks

    # ----------------------------------------------------------------------
    # Has Data
    # ----------------------------------------------------------------------
    def _collection_has_data(self):
        try:
            resp = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                limit=1,
            )
            return bool(resp)
        except Exception:
            return False

    # ----------------------------------------------------------------------
    # Insert Data
    # ----------------------------------------------------------------------
    def _insert_data_if_needed(self):
        if not self.documents:
            logger.info("No documents to insert.")
            return

        if self._collection_has_data():
            logger.info("Collection already contains data. Skipping insert.")
            return

        logger.info("Validating vectors...")

        clean_docs = []
        for idx, d in enumerate(self.documents):
            vec = d["vector"]

            # Fix [[...]] nested lists
            if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
                vec = vec[0]

            if len(vec) != self.dim:
                raise ValueError(f"Vector {idx} has incorrect dimension {len(vec)}")

            if not all(isinstance(x, (float, int)) for x in vec):
                raise ValueError(f"Vector {idx} contains non-float elements")

            clean_docs.append({
                "vector": vec,
                "content": d["content"],
                "source": d["source"],
                "type": d["type"],
                "chunk_number": d["chunk_number"],
            })

        logger.info("Inserting %d rows...", len(clean_docs))

        insert_result = self.client.insert(
            collection_name=self.collection_name,
            data=clean_docs
        )

        self.client.flush(self.collection_name)

        logger.info("Insert complete: %s", insert_result)

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------
    def search(self, query, top_k=5):
        q_emb = self.model.encode([query]).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            data=q_emb,
            limit=top_k,
            output_fields=["content", "source", "type", "chunk_number"],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            index_name=""
        )

        formatted = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            formatted.append({
                "content": entity.get("content"),
                "source": entity.get("source"),
                "type": entity.get("type"),
                "chunk_number": entity.get("chunk_number"),
                "distance": hit.get("distance"),
            })

        return formatted

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------
    def reset_collection(self):
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
        except Exception:
            pass

        self._make_local_collection()
