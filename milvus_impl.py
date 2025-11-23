from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import os
import re
import glob
import logging
import streamlit as st

class VectorStore:
    def __init__(self):
        # init config for model and milvus
        # bare-bones
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = MilvusClient("./ezrag_local.db")
        self.collection_name = "ezrag_collection"
        
        self.documents = self._prepare_documents()

        self.collection = self._make_local_collection()

        if self.documents:
            self._insert_data()

    def _make_local_collection(self):
        st.info(f"Making Collection '{self.collection_name}'")
        if not self.client.has_collection(self.collection_name):
            try:
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="chunk_number", dtype=DataType.INT64),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
                ]

                schema = CollectionSchema(fields, description="ezrag vector store")

                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                )

                st.success(f"Collection '{self.collection_name}' created successfully.")
            except Exception as e:
                st.error(f"Error creating collection: {str(e)}")
                raise
        else:
            logging.error(f"Collection '{self.collection_name}' already exists.")

    def _prepare_documents(self):
        documents = []
        all_chunks = []
        documents_dir = "documents"  # push to config
        markdown_files = glob.glob(os.path.join(documents_dir, "*.md"))
        # Extract and Split
        for file_path in markdown_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Extract document type from filename (without extension)
                doc_type = os.path.splitext(filename)[0]

                # Split content into chunks based on markdown headers
                chunks = self._split_markdown_into_chunks(content, filename)

                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        all_chunks.append(
                            {
                                "content": chunk.strip(),
                                "source": filename,
                                "type": doc_type,
                                "chunk_number": i + 1,
                            }
                        )
            except Exception as e:
                logging.error(f"Error reading {filename}: {str(e)}")

        # Vectorize and Format
        if all_chunks:
            chunk_contents = [chunk["content"] for chunk in all_chunks]
            embeddings = self.model.encode(chunk_contents).tolist()

            # Prepare documents with embeddings and IDs for Milvus
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                documents.append(
                    {
                        "vector": embedding,
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "type": chunk["type"],
                        "chunk_number": chunk["chunk_number"],
                    }
                )

        return documents

    def _split_markdown_into_chunks(self, content, filename):
        chunks = []

        # Split by markdown headers (###, ##, #)
        header_pattern = r"(?=\n#{1,3}\s+)"
        sections = re.split(header_pattern, content)

        for section in sections:
            if section.strip():
                # Clean up the section and add to chunks
                cleaned_section = section.strip()
                chunks.append(cleaned_section)

        # If no headers found, split by paragraphs
        if len(chunks) <= 1:
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

        # If still too few chunks, split by sentences for very small files
        if len(chunks) <= 1:
            sentences = re.split(r"[.!?]+", content)
            chunks = [
                sentence.strip() for sentence in sentences if len(sentence.strip()) > 50
            ]

        return chunks

    def _insert_data(self):
        if not self.documents:
            logging.error("No documents to insert!")
            return

        existing_count = self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",  # Simple filter to check if any data exists
            limit=1,
        )

        if existing_count:
            logging.info("Data already exists in the collection. Skipping insertion.")
            return

        # Insert data
        insert_result = self.client.insert(
            collection_name=self.collection_name, data=self.documents
        )
        logging.info(f"Inserted {len(insert_result)}  Document Chunks")

    def search(self, query, top_k=5):
        if not self.documents:
            return []

        # Encode query
        query_embedding = self.model.encode([query]).tolist()

        # Perform the search
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_embedding,
            limit=top_k,
            output_fields=[
                "content",
                "source",
                "type",
                "chunk_number",
            ],  # Fields you want returned
        )

        # The results are already a list of hits for the single query.
        formatted_results = []
        for hit in results[0]:  # Access the first (and only) list of results
            formatted_results.append(
                {
                    "content": hit["entity"].get("content"),
                    "source": hit["entity"].get("source"),
                    "type": hit["entity"].get("type"),
                    "chunk_number": hit["entity"].get("chunk_number"),
                    "distance": hit["distance"],
                }
            )

        return formatted_results
