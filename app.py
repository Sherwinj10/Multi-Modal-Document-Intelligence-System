import streamlit as st
import os
import shutil
from src.ingestion import load_documents
from src.retrieval import build_index, load_index, get_retriever
from src.generation import get_query_engine, generate_response
from src.utils import load_env_vars

# Load environment variables
load_env_vars()

st.set_page_config(page_title="Multi-Modal RAG", layout="wide")

st.title("📄 Multi-Modal Document Intelligence")
st.markdown("Upload documents (PDFs) and ask questions about them. Handles text, tables, and images.")

# Sidebar for configuration and file upload
with st.sidebar:
    st.header("Configuration")
    
    # API Keys check
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in .env")
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        st.warning("LLAMA_CLOUD_API_KEY not found. LlamaParse might fail.")

    st.header("Data Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files to data directory
                data_dir = "data"
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(data_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Ingest and Build Index
                try:
                    documents = load_documents(data_dir)
                    st.session_state.index = build_index(documents)
                    st.success(f"Processed {len(documents)} documents and built index!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        else:
            st.warning("Please upload files first.")

    if st.button("Clear Index"):
        if os.path.exists("storage"):
            shutil.rmtree("storage")
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
        if "index" in st.session_state:
            del st.session_state.index
        st.success("Index cleared!")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load index if exists and not loaded
if "index" not in st.session_state:
    if os.path.exists("storage"):
        try:
            st.session_state.index = load_index()
            st.info("Loaded existing index.")
        except Exception as e:
            st.error(f"Error loading index: {e}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if "index" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query_engine = get_query_engine(st.session_state.index)
                    response = generate_response(query_engine, prompt)
                    
                    st.markdown(response.response)
                    
                    # Display sources
                    with st.expander("View Sources"):
                        for node in response.source_nodes:
                            st.markdown(f"**Score:** {node.score:.2f}")
                            st.markdown(f"**Content:** {node.node.get_content()[:200]}...")
                            st.markdown("---")
                            
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.warning("Please process documents first.")
