import streamlit as st
import os
from glob import glob
from milvus_impl import VectorStore

def main():
    st.title("Simple Document RAG System")
    st.write("Ask questions about your document collection!")
    
    # Initialize RAG system
    if 'rag' not in st.session_state:
        with st.spinner("Loading RAG system..."):
            st.session_state.rag = VectorStore()
    
    # Show document statistics
    if st.session_state.rag.documents:
        st.sidebar.header("Document Stats")
        st.sidebar.write(f"Total chunks: {len(st.session_state.rag.documents)}")
        
        # Count by type
        type_count = {}
        for doc in st.session_state.rag.documents:
            doc_type = doc['type']
            type_count[doc_type] = type_count.get(doc_type, 0) + 1
        
        for doc_type, count in type_count.items():
            st.sidebar.write(f"- {doc_type}: {count} chunks")
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="e.g., What are the main topics in the documents?")
    
    if query:
        with st.spinner("Searching through documents..."):
            results = st.session_state.rag.search(query, top_k=5)
        
        if results:
            st.subheader(f" Found {len(results)} relevant chunks:")
            
            for i, result in enumerate(results, 1):
                with st.expander(f" {result['source']} - Chunk {result['chunk_number']} (Relevance: {1/(1+result['distance']):.2%})"):
                    st.write(result['content'])
                    
                    # Add metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"**Source:** {result['source']}")
                    with col2:
                        st.caption(f"**Type:** {result['type']}")
                    with col3:
                        st.caption(f"**Chunk:** {result['chunk_number']}")
        else:
            st.warning("No relevant documents found. Try a different query.")
    
    # File management section
    st.sidebar.header("Document Management")
    st.sidebar.write("Add .md files to the 'documents' directory")
    
    # Show current files
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        current_files = glob.glob(os.path.join(documents_dir, "*.md"))
        if current_files:
            st.sidebar.write("**Current files:**")
            for file in current_files:
                st.sidebar.write(f"- {os.path.basename(file)}")
        else:
            st.sidebar.write("No markdown files found.")
    
    # Refresh button
    if st.sidebar.button("Refresh Documents"):
        st.session_state.pop('rag', None)
        st.rerun()
    
    # Example queries
    st.sidebar.header("💡 Example Queries")
    example_queries = [
        "What are the main topics?",
        "Summarize the key points",
        "Find interesting facts",
        "What technology is discussed?",
        "List historical events mentioned"
    ]
    
    for example in example_queries:
        if st.sidebar.button(example):
            st.experimental_set_query_params(query=example)
            st.rerun()

if __name__=="__main__":
    main()
