import streamlit as st
from chroma_impl import VectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Simple Document RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_store():
    # VectorStore initialization may be slow (model load). Cache it across reruns.
    return VectorStore(
        documents_dir="documents",
        persist_directory="chroma_store",
        collection_name="ezrag_collection",
    )


def main():
    st.title("EZRAG")
    st.write("Place markdown files in the `documents/` folder. The app will read, chunk, embed and store them locally (first run).")

    with st.spinner("Loading vector collection (first run may take a minute)..."):
        store = get_store()

    # Sidebar controls
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Top K results", min_value=1, max_value=20, value=5, step=1)
    refresh = st.sidebar.button("Re-ingest documents (drop & rebuild collection)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Documents folder: `./documents/`")

    if refresh:
        with st.spinner("Resetting collection and re-ingesting..."):
            store.reset_collection()
        st.success("Re-ingest complete.")
        st.rerun()

    # Query input
    query = st.text_input("Enter your question:", placeholder="e.g., What are the main topics?")

    if query:
        with st.spinner("Searching..."):
            results = store.search(query, top_k=top_k)

        if results:
            st.subheader(f"Found {len(results)} chunks (top_k={top_k}):")
            for i, r in enumerate(results, start=1):
                # compute a simple similarity percentage if distance is available
                similarity = None
                if r.get("distance") is not None:
                    try:
                        similarity = 1 / (1 + float(r["distance"]))
                    except Exception:
                        similarity = None

                header = f"{r.get('source', 'unknown')} — chunk {r.get('chunk_number', '?')}"
                if similarity is not None:
                    header += f" — relevance: {similarity:.2%}"

                with st.expander(header):
                    st.write(r.get("content", ""))
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.caption(f"Source: {r.get('source')}")
                    with c2:
                        st.caption(f"Type: {r.get('type')}")
                    with c3:
                        st.caption(f"Chunk: {r.get('chunk_number')}")
        else:
            st.info("No results found. Try increasing Top K or re-ingest if you added new documents.")


if __name__ == "__main__":
    main()
