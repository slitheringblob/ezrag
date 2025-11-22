from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
import re
from glob import glob
import logging

class VectorStore:
    def __init__(self):
        # init config for model and milvus
        # bare-bones
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        connections.connect('default', host="localhost", port="19530")
        
        # split text to vectorize
        self.documents = self._prepare_documents()
        
        # Create collection
        self.collection = self._create_collection()
        
        # Insert data if collection is empty
        if self.collection.is_empty:
            self._insert_data()
    
    def _prepare_documents(self):
        documents = []
        documents_dir = "documents" #push to config
        markdown_files = glob.glob(os.path.join(documents_dir, "*.md"))
        
        for file_path in markdown_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Extract document type from filename (without extension)
                doc_type = os.path.splitext(filename)[0]
                
                # Split content into chunks based on markdown headers
                chunks = self._split_markdown_into_chunks(content, filename)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append({
                            'content': chunk.strip(),
                            'source': filename,
                            'type': doc_type,
                            'chunk_number': i + 1
                        })
            except Exception as e:
                logging.error(f"Error reading {filename}: {str(e)}")
        return documents

    def _split_markdown_into_chunks(self, content, filename):

        chunks = []
        
        # Split by markdown headers (###, ##, #)
        header_pattern = r'(?=\n#{1,3}\s+)'
        sections = re.split(header_pattern, content)
        
        for section in sections:
            if section.strip():
                # Clean up the section and add to chunks
                cleaned_section = section.strip()
                chunks.append(cleaned_section)
        
        # If no headers found, split by paragraphs
        if len(chunks) <= 1:
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        # If still too few chunks, split by sentences for very small files
        if len(chunks) <= 1:
            sentences = re.split(r'[.!?]+', content)
            chunks = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 50]

        return chunks

    def _create_collection(self):
        collection_name = "ezrag" #push to config
        
        # Delete collection if exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        # Define fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_number", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384) #mini lm v2 dims
        ]
        
        schema = CollectionSchema(fields = fields, description="ezrag collection contains cool facts")
        collection = Collection(name = collection_name, schema = schema)
        
        # Create index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        return collection

    def _insert_data(self):
        if not self.documents:
            logging.error("No documents to insert!")
            return
            
        contents = [doc['content'] for doc in self.documents]
        sources = [doc['source'] for doc in self.documents]
        types = [doc['type'] for doc in self.documents]
        chunk_numbers = [doc['chunk_number'] for doc in self.documents]
        
        # Generate embeddings
        embeddings = self.model.encode(contents).tolist()
        
        # Prepare data for insertion
        entities = [
            contents,
            sources,
            types, 
            chunk_numbers,
            embeddings
        ]
        
        # Insert data
        self.collection.insert(entities)
        self.collection.load()
        logging.info(f"Successfully inserted {len(self.documents)} document chunks into the !")

    def search(self, query, top_k=5):
        if not self.documents:
            return []
            
        # Encode query
        query_embedding = self.model.encode([query]).tolist()
        
        # Search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Perform search
        results = self.collection.search(
            query_embedding, 
            "embedding", 
            search_params, 
            limit=top_k,
            output_fields=["content", "source", "type", "chunk_number"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'content': hit.entity.get('content'),
                    'source': hit.entity.get('source'),
                    'type': hit.entity.get('type'),
                    'chunk_number': hit.entity.get('chunk_number'),
                    'distance': hit.distance
                })
        return formatted_results